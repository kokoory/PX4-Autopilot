#!/usr/bin/env python3
"""
Export trained PyTorch Student MLP to TFLite format for PX4 deployment.

Pipeline: PyTorch (.pt) -> ONNX (.onnx) -> TFLite (.tflite) -> C++ byte array

The C++ output replaces control_net.cpp/hpp in mc_nn_control module.

Usage:
    python export_tflite.py --model output/student.pt --config model_config.yaml
    python export_tflite.py --model output/student.pt --output ../../src/modules/mc_nn_control/
"""

import argparse
import logging
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from train_student import StudentMLP, create_model_from_config

logger = logging.getLogger(__name__)


def export_to_onnx(model: StudentMLP, onnx_path: str, input_size: int = 15):
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["motor_command"],
        dynamic_axes=None,  # Fixed batch size for embedded
    )

    logger.info(f"Exported ONNX model to {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, quantization: str = "none"):
    """Convert ONNX model to TFLite format."""
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        # ONNX -> TF SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_dir = os.path.join(tmpdir, "saved_model")
            tf_rep.export_graph(saved_model_dir)

            # TF SavedModel -> TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

            if quantization == "dynamic_range":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization == "full_integer":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]

            tflite_model = converter.convert()

    except ImportError:
        logger.info("onnx_tf not available, trying tf2onnx + tflite-runtime approach")
        tflite_model = _convert_via_tf2onnx(onnx_path, quantization)

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    logger.info(f"Exported TFLite model to {tflite_path} ({len(tflite_model)} bytes)")
    return len(tflite_model)


def _convert_via_tf2onnx(onnx_path: str, quantization: str) -> bytes:
    """Alternative conversion path using tf2onnx command line."""
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")

        # Use tf2onnx to convert ONNX -> TF SavedModel
        result = subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--onnx", onnx_path,
                "--output", os.path.join(tmpdir, "model.onnx"),
                "--opset", "13",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Direct approach: load ONNX, manually create TF model
            return _convert_manual(onnx_path, quantization)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if quantization == "dynamic_range":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        return converter.convert()


def _convert_manual(onnx_path: str, quantization: str) -> bytes:
    """Manual conversion: reconstruct TF model from ONNX weights."""
    import onnx
    import tensorflow as tf

    onnx_model = onnx.load(onnx_path)

    # Extract weights from ONNX
    weights = {}
    for initializer in onnx_model.graph.initializer:
        np_array = np.frombuffer(initializer.raw_data, dtype=np.float32)
        np_array = np_array.reshape(list(initializer.dims))
        weights[initializer.name] = np_array

    # Build equivalent TF model from ONNX graph structure
    # Parse the sequential structure
    layers = []
    for node in onnx_model.graph.node:
        if node.op_type == "MatMul" or node.op_type == "Gemm":
            layers.append(("dense", node))
        elif node.op_type == "Relu":
            layers.append(("relu", node))
        elif node.op_type == "Add":
            layers.append(("add", node))

    # Create TF model
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    input_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]

    tf_input = tf.keras.Input(shape=(input_size,), name="observation")
    x = tf_input

    for layer_type, node in layers:
        if layer_type == "dense":
            # Find weight and bias
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None

            if weight_name and weight_name in weights:
                w = weights[weight_name]
                b = weights.get(bias_name, np.zeros(w.shape[-1]))
                dense = tf.keras.layers.Dense(
                    w.shape[-1],
                    use_bias=True,
                    weights=[w.T if w.ndim == 2 else w, b],
                )
                x = dense(x)
        elif layer_type == "relu":
            x = tf.keras.layers.ReLU()(x)

    tf_model = tf.keras.Model(inputs=tf_input, outputs=x)

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    if quantization == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()


def generate_cc_arrays(tflite_path: str, output_dir: str, model_name: str = "control_net"):
    """Generate C++ source files containing the TFLite model as byte arrays."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    model_size = len(data)

    # Generate .hpp header
    hpp_content = f"""#include <cstdint>

constexpr unsigned int {model_name}_tflite_size = {model_size};
extern const unsigned char {model_name}_tflite[];
"""

    # Generate .cpp source with byte array
    cpp_lines = [
        '#include <cstdint>',
        f'#include "{model_name}.hpp"',
        '',
        f'alignas(16) const unsigned char {model_name}_tflite[] = {{',
    ]

    # Format bytes as hex, 12 per line (matching existing PX4 style)
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + 12 < len(data):
            cpp_lines.append(f"\t{hex_values},")
        else:
            cpp_lines.append(f"\t{hex_values}")

    cpp_lines.append("};")
    cpp_lines.append("")

    hpp_path = os.path.join(output_dir, f"{model_name}.hpp")
    cpp_path = os.path.join(output_dir, f"{model_name}.cpp")

    with open(hpp_path, "w") as f:
        f.write(hpp_content)

    with open(cpp_path, "w") as f:
        f.write("\n".join(cpp_lines))

    logger.info(f"Generated {hpp_path} ({model_size} bytes)")
    logger.info(f"Generated {cpp_path}")

    return hpp_path, cpp_path, model_size


def verify_tflite_model(tflite_path: str, pytorch_model: StudentMLP, input_size: int = 15):
    """Verify TFLite model produces same outputs as PyTorch model."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow as tf
            tflite = tf.lite
        except ImportError:
            logger.warning("No TFLite runtime available for verification")
            return True

    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test with random inputs
    pytorch_model.eval()
    max_error = 0.0
    num_tests = 100

    for _ in range(num_tests):
        test_input = np.random.randn(1, input_size).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            pt_output = pytorch_model(torch.FloatTensor(test_input)).numpy()

        # TFLite inference
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]["index"])

        error = np.max(np.abs(pt_output - tflite_output))
        max_error = max(max_error, error)

    logger.info(f"Max PyTorch vs TFLite error: {max_error:.8f}")
    return max_error


def main():
    parser = argparse.ArgumentParser(description="Export student MLP to TFLite for PX4")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch .pt model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "model_config.yaml"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for C++ files (default: mc_nn_control/)",
    )
    parser.add_argument("--quantization", type=str, default=None, choices=["none", "dynamic_range", "full_integer"])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    export_config = config.get("export", {})
    quantization = args.quantization or export_config.get("quantization", "none")
    max_size = export_config.get("max_model_size_bytes", 20000)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = export_config.get(
            "output_dir",
            str(Path(__file__).parent / ".." / ".." / "src" / "modules" / "mc_nn_control"),
        )
        # Resolve relative path from config file location
        if not os.path.isabs(output_dir):
            output_dir = str((Path(__file__).parent / output_dir).resolve())

    # Load model
    model = create_model_from_config(config)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    logger.info(f"Loaded model with {model.count_parameters()} parameters")

    # Create intermediate directory
    intermediate_dir = str(Path(__file__).parent / "output")
    os.makedirs(intermediate_dir, exist_ok=True)

    # Step 1: PyTorch -> ONNX
    onnx_path = os.path.join(intermediate_dir, "student.onnx")
    export_to_onnx(model, onnx_path, config["model"]["input_size"])

    # Step 2: ONNX -> TFLite
    tflite_path = os.path.join(intermediate_dir, "student.tflite")
    model_size = convert_onnx_to_tflite(onnx_path, tflite_path, quantization)

    if model_size > max_size:
        logger.warning(
            f"Model size ({model_size} bytes) exceeds limit ({max_size} bytes). "
            f"Consider reducing hidden layers or using quantization."
        )

    # Step 3: Verify TFLite output matches PyTorch
    max_error = verify_tflite_model(tflite_path, model, config["model"]["input_size"])
    validation_config = config.get("validation", {})
    error_threshold = validation_config.get("max_output_error", 0.001)

    if isinstance(max_error, float) and max_error > error_threshold:
        logger.warning(
            f"TFLite conversion error ({max_error:.6f}) exceeds threshold ({error_threshold}). "
            f"Consider checking the conversion pipeline."
        )

    # Step 4: Generate C++ byte arrays
    os.makedirs(output_dir, exist_ok=True)
    hpp_path, cpp_path, size = generate_cc_arrays(tflite_path, output_dir)

    print(f"\nExport complete:")
    print(f"  ONNX:   {onnx_path}")
    print(f"  TFLite: {tflite_path} ({model_size} bytes)")
    print(f"  C++ header: {hpp_path}")
    print(f"  C++ source: {cpp_path}")
    print(f"  Model size: {size} bytes (limit: {max_size})")
    if isinstance(max_error, float):
        print(f"  Max conversion error: {max_error:.8f}")


if __name__ == "__main__":
    main()
