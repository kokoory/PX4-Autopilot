#!/usr/bin/env python3
"""
Validate exported TFLite model for PX4 deployment.

Checks:
1. Output correctness (PyTorch vs TFLite comparison)
2. Output range validation ([-1, 1])
3. Model size constraints
4. Inference time benchmarking
5. Numerical stability with edge-case inputs

Usage:
    python validate_model.py --tflite output/student.tflite --pytorch output/student.pt
    python validate_model.py --tflite output/student.tflite --config model_config.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from train_student import create_model_from_config

logger = logging.getLogger(__name__)


def load_tflite_interpreter(tflite_path: str):
    """Load TFLite interpreter from file."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass

    raise ImportError("Neither tflite_runtime nor tensorflow is available")


def run_tflite_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run inference on TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def validate_output_correctness(
    tflite_path: str,
    pytorch_model: torch.nn.Module,
    num_samples: int = 1000,
    input_size: int = 15,
) -> dict:
    """Compare TFLite and PyTorch outputs."""
    interpreter = load_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()

    pytorch_model.eval()
    errors = []

    for _ in range(num_samples):
        test_input = np.random.randn(1, input_size).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            pt_output = pytorch_model(torch.FloatTensor(test_input)).numpy()

        # TFLite
        tflite_output = run_tflite_inference(interpreter, test_input)

        error = np.abs(pt_output - tflite_output)
        errors.append(error)

    errors = np.array(errors)
    return {
        "max_absolute_error": float(np.max(errors)),
        "mean_absolute_error": float(np.mean(errors)),
        "p99_absolute_error": float(np.percentile(errors, 99)),
        "num_samples": num_samples,
    }


def validate_output_range(
    tflite_path: str,
    num_samples: int = 1000,
    input_size: int = 15,
    expected_range: tuple = (-1.0, 1.0),
) -> dict:
    """Check that model outputs are within expected range."""
    interpreter = load_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()

    out_of_range_count = 0
    max_output = float("-inf")
    min_output = float("inf")

    # Test with various input distributions
    test_inputs = [
        # Normal inputs
        np.random.randn(1, input_size).astype(np.float32),
        # Large inputs
        np.random.randn(1, input_size).astype(np.float32) * 10.0,
        # Small inputs
        np.random.randn(1, input_size).astype(np.float32) * 0.01,
        # Zero input
        np.zeros((1, input_size), dtype=np.float32),
        # Ones input
        np.ones((1, input_size), dtype=np.float32),
    ]

    for _ in range(num_samples):
        test_input = np.random.randn(1, input_size).astype(np.float32)
        test_inputs.append(test_input)

    for test_input in test_inputs:
        output = run_tflite_inference(interpreter, test_input)
        max_output = max(max_output, float(np.max(output)))
        min_output = min(min_output, float(np.min(output)))

        if np.any(output < expected_range[0] - 0.1) or np.any(output > expected_range[1] + 0.1):
            out_of_range_count += 1

    return {
        "min_output": min_output,
        "max_output": max_output,
        "out_of_range_count": out_of_range_count,
        "total_samples": len(test_inputs),
        "expected_range": list(expected_range),
    }


def validate_numerical_stability(
    tflite_path: str, input_size: int = 15
) -> dict:
    """Test model with edge-case inputs for numerical stability."""
    interpreter = load_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()

    edge_cases = {
        "zeros": np.zeros((1, input_size), dtype=np.float32),
        "large_positive": np.full((1, input_size), 100.0, dtype=np.float32),
        "large_negative": np.full((1, input_size), -100.0, dtype=np.float32),
        "mixed_extreme": np.array([[100, -100, 0, 1, 0, 0, 0, 1, 0, 50, -50, 0, 10, -10, 0]], dtype=np.float32),
        "nan_free_check": np.random.randn(1, input_size).astype(np.float32) * 5.0,
    }

    results = {}
    all_finite = True

    for name, test_input in edge_cases.items():
        output = run_tflite_inference(interpreter, test_input)
        is_finite = bool(np.all(np.isfinite(output)))
        all_finite = all_finite and is_finite

        results[name] = {
            "output": output.flatten().tolist(),
            "is_finite": is_finite,
            "max_abs": float(np.max(np.abs(output))) if is_finite else float("inf"),
        }

    return {
        "all_outputs_finite": all_finite,
        "edge_case_results": results,
    }


def validate_model_size(tflite_path: str, max_size_bytes: int = 20000) -> dict:
    """Verify model size is within deployment constraints."""
    file_size = os.path.getsize(tflite_path)
    return {
        "model_size_bytes": file_size,
        "max_allowed_bytes": max_size_bytes,
        "within_limit": file_size <= max_size_bytes,
        "utilization_pct": round(file_size / max_size_bytes * 100, 1),
    }


def benchmark_inference(
    tflite_path: str,
    num_iterations: int = 100,
    input_size: int = 15,
    target_time_us: int = 2500,
) -> dict:
    """Benchmark inference time on host (indicative, not target hardware)."""
    interpreter = load_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()

    test_input = np.random.randn(1, input_size).astype(np.float32)

    # Warmup
    for _ in range(10):
        run_tflite_inference(interpreter, test_input)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        run_tflite_inference(interpreter, test_input)
        elapsed_us = (time.perf_counter() - start) * 1e6
        times.append(elapsed_us)

    times = np.array(times)
    return {
        "mean_inference_us": float(np.mean(times)),
        "median_inference_us": float(np.median(times)),
        "p99_inference_us": float(np.percentile(times, 99)),
        "max_inference_us": float(np.max(times)),
        "min_inference_us": float(np.min(times)),
        "target_time_us": target_time_us,
        "num_iterations": num_iterations,
        "note": "Host benchmark only - embedded performance will differ",
    }


def run_full_validation(
    tflite_path: str,
    pytorch_model_path: str = None,
    config: dict = None,
) -> dict:
    """Run all validation checks and return combined report."""
    results = {"tflite_path": tflite_path, "passed": True, "checks": {}}

    if config is None:
        config = {}

    validation_config = config.get("validation", {})
    export_config = config.get("export", {})
    model_config = config.get("model", {})
    input_size = model_config.get("input_size", 15)

    # 1. Model size
    logger.info("Checking model size...")
    size_result = validate_model_size(
        tflite_path,
        export_config.get("max_model_size_bytes", 20000),
    )
    results["checks"]["model_size"] = size_result
    if not size_result["within_limit"]:
        results["passed"] = False
        logger.error(f"FAIL: Model size {size_result['model_size_bytes']} > {size_result['max_allowed_bytes']}")
    else:
        logger.info(f"PASS: Model size {size_result['model_size_bytes']} bytes ({size_result['utilization_pct']}%)")

    # 2. Output correctness (if PyTorch model available)
    if pytorch_model_path and config:
        logger.info("Checking output correctness...")
        model = create_model_from_config(config)
        model.load_state_dict(torch.load(pytorch_model_path, weights_only=True))

        correctness = validate_output_correctness(
            tflite_path, model,
            num_samples=validation_config.get("num_test_samples", 1000),
            input_size=input_size,
        )
        results["checks"]["output_correctness"] = correctness

        threshold = validation_config.get("max_output_error", 0.001)
        if correctness["max_absolute_error"] > threshold:
            results["passed"] = False
            logger.error(f"FAIL: Max error {correctness['max_absolute_error']:.6f} > {threshold}")
        else:
            logger.info(f"PASS: Max error {correctness['max_absolute_error']:.6f}")

    # 3. Output range
    logger.info("Checking output range...")
    output_range = validation_config.get("output_range", [-1.0, 1.0])
    range_result = validate_output_range(
        tflite_path, input_size=input_size,
        expected_range=tuple(output_range),
    )
    results["checks"]["output_range"] = range_result
    if range_result["out_of_range_count"] > 0:
        logger.warning(
            f"WARNING: {range_result['out_of_range_count']} samples out of range "
            f"[{range_result['min_output']:.4f}, {range_result['max_output']:.4f}]"
        )
    else:
        logger.info(f"PASS: Outputs in range [{range_result['min_output']:.4f}, {range_result['max_output']:.4f}]")

    # 4. Numerical stability
    logger.info("Checking numerical stability...")
    stability = validate_numerical_stability(tflite_path, input_size=input_size)
    results["checks"]["numerical_stability"] = stability
    if not stability["all_outputs_finite"]:
        results["passed"] = False
        logger.error("FAIL: Non-finite outputs detected")
    else:
        logger.info("PASS: All edge-case outputs are finite")

    # 5. Inference benchmark
    logger.info("Benchmarking inference time...")
    benchmark = benchmark_inference(
        tflite_path,
        num_iterations=validation_config.get("benchmark_iterations", 100),
        input_size=input_size,
        target_time_us=export_config.get("target_inference_time_us", 2500),
    )
    results["checks"]["inference_benchmark"] = benchmark
    logger.info(
        f"Inference: mean={benchmark['mean_inference_us']:.1f}us, "
        f"p99={benchmark['p99_inference_us']:.1f}us "
        f"(target: {benchmark['target_time_us']}us, host benchmark)"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate TFLite model for PX4 deployment")
    parser.add_argument("--tflite", type=str, required=True, help="Path to TFLite model")
    parser.add_argument("--pytorch", type=str, default=None, help="Path to PyTorch .pt model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "model_config.yaml"),
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = run_full_validation(
        tflite_path=args.tflite,
        pytorch_model_path=args.pytorch,
        config=config,
    )

    # Save report
    if args.output:
        report_path = args.output
    else:
        report_path = str(Path(args.tflite).parent / "validation_report.json")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Validation {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"{'=' * 60}")
    for check_name, check_result in results["checks"].items():
        status = "OK"
        if check_name == "model_size" and not check_result.get("within_limit", True):
            status = "FAIL"
        elif check_name == "numerical_stability" and not check_result.get("all_outputs_finite", True):
            status = "FAIL"
        print(f"  [{status:>4}] {check_name}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
