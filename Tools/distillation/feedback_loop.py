#!/usr/bin/env python3
"""
Closed-Loop Runtime Distillation Feedback Orchestrator.

Manages the full cycle:
  1. Train initial MLP from LLM-generated reward (Eureka-style)
  2. Export to TFLite and deploy to PX4
  3. Fly and collect DistillationStatus data
  4. Analyze flight data for sim-to-real gap
  5. LLM reasons about gap and generates improved reward
  6. Retrain MLP with improved reward
  7. Repeat until convergence or max iterations

This is the entry point for the complete pipeline.

Usage:
    python feedback_loop.py --config model_config.yaml --scenarios scenarios.yaml
    python feedback_loop.py --flight-log flight.ulg --iteration 2
    python feedback_loop.py --run-full --max-cycles 5
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from reward_generator import run_pipeline as generate_rewards
from train_student import create_model_from_config, train_behavior_cloning, train_rl_distillation
from flight_analyzer import analyze_flight, format_report_for_llm
from sim_real_bridge import generate_sim_real_reward

logger = logging.getLogger(__name__)


class DistillationFeedbackLoop:
    """
    Orchestrates the closed-loop distillation pipeline.

    Tracks state across cycles:
    - Which reward version is active
    - Performance history across iterations
    - Convergence metrics
    """

    def __init__(
        self,
        config_path: str,
        scenarios_path: str,
        workspace_dir: str = None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open(scenarios_path) as f:
            self.scenarios = yaml.safe_load(f)

        self.config_path = config_path
        self.scenarios_path = scenarios_path

        if workspace_dir is None:
            workspace_dir = str(Path(config_path).parent / "workspace")

        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

        self.state_file = os.path.join(workspace_dir, "loop_state.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)

        return {
            "current_cycle": 0,
            "cycles": [],
            "converged": False,
            "best_cycle": None,
            "best_position_error": float("inf"),
        }

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def run_initial_training(self, scenario_name: str = "hover_stable") -> dict:
        """
        Cycle 0: Generate initial reward with LLM and train first MLP.
        This is the Eureka-equivalent step.
        """
        cycle_dir = os.path.join(self.workspace_dir, "cycle_0")
        os.makedirs(cycle_dir, exist_ok=True)

        logger.info("=== Cycle 0: Initial reward generation (Eureka-style) ===")

        # Step 1: Generate reward functions
        reward_results = generate_rewards(
            config_path=self.config_path,
            scenarios_path=self.scenarios_path,
            scenario_name=scenario_name,
            output_dir=os.path.join(cycle_dir, "rewards"),
        )

        reward_path = list(reward_results.values())[0]["reward_path"]

        # Step 2: Train student MLP
        logger.info("Training initial student MLP...")
        model = create_model_from_config(self.config)
        training_mode = self.config.get("training", {}).get("mode", "behavior_cloning")

        model_dir = os.path.join(cycle_dir, "model")

        if training_mode == "rl_distillation":
            metrics = train_rl_distillation(
                model, self.config, reward_path, model_dir
            )
        else:
            metrics = train_behavior_cloning(
                model, self.config, output_dir=model_dir
            )

        cycle_result = {
            "cycle": 0,
            "type": "initial",
            "reward_path": reward_path,
            "model_path": os.path.join(model_dir, "student.pt"),
            "training_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.state["current_cycle"] = 0
        self.state["cycles"].append(cycle_result)
        self._save_state()

        logger.info(f"Cycle 0 complete. Model: {cycle_result['model_path']}")
        return cycle_result

    def run_feedback_cycle(
        self,
        flight_log_path: str,
        sim_baseline_path: str = None,
    ) -> dict:
        """
        Run one feedback cycle:
        1. Analyze flight data
        2. LLM identifies sim-to-real gap
        3. Generate improved reward
        4. Retrain MLP
        """
        cycle_num = self.state["current_cycle"] + 1
        cycle_dir = os.path.join(self.workspace_dir, f"cycle_{cycle_num}")
        os.makedirs(cycle_dir, exist_ok=True)

        logger.info(f"=== Cycle {cycle_num}: Feedback refinement ===")

        # Step 1: Analyze flight data
        logger.info("Step 1: Analyzing flight data...")
        sim_baseline = None
        if sim_baseline_path:
            with open(sim_baseline_path) as f:
                sim_baseline = json.load(f)

        report = analyze_flight(flight_log_path, sim_baseline)
        report_text = format_report_for_llm(report)

        report_path = os.path.join(cycle_dir, "flight_report.json")
        from dataclasses import asdict
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

        report_prompt_path = os.path.join(cycle_dir, "flight_report_llm.md")
        with open(report_prompt_path, "w") as f:
            f.write(report_text)

        logger.info(f"Flight report: pos_error={report.overall_mean_position_error:.3f}m, "
                     f"success_rate={report.overall_success_rate:.2%}")

        # Step 2: Get current reward from previous cycle
        prev_cycle = self.state["cycles"][-1]
        current_reward_path = prev_cycle["reward_path"]

        # Step 3: LLM sim-to-real analysis and reward improvement
        logger.info("Step 2: LLM sim-to-real gap analysis...")
        vehicle_params = self.scenarios.get("vehicle_defaults", {})

        sim_real_results = generate_sim_real_reward(
            flight_report_path=report_path,
            current_reward_path=current_reward_path,
            vehicle_params=vehicle_params,
            sim_baseline_path=sim_baseline_path,
            output_dir=os.path.join(cycle_dir, "sim_real"),
            iterations=1,  # Single LLM iteration per feedback cycle
        )

        new_reward_path = sim_real_results["final_reward_path"]

        # Step 4: Retrain MLP with improved reward
        logger.info("Step 3: Retraining MLP with sim-to-real refined reward...")
        model = create_model_from_config(self.config)
        model_dir = os.path.join(cycle_dir, "model")
        training_mode = self.config.get("training", {}).get("mode", "behavior_cloning")

        if training_mode == "rl_distillation":
            metrics = train_rl_distillation(
                model, self.config, new_reward_path, model_dir
            )
        else:
            metrics = train_behavior_cloning(
                model, self.config, output_dir=model_dir
            )

        # Track improvement
        current_error = report.overall_mean_position_error
        improved = current_error < self.state["best_position_error"]

        if improved:
            self.state["best_position_error"] = current_error
            self.state["best_cycle"] = cycle_num

        # Check convergence
        convergence_threshold = 0.1  # meters
        if current_error < convergence_threshold:
            self.state["converged"] = True
            logger.info(f"CONVERGED: Position error {current_error:.3f}m < {convergence_threshold}m")

        cycle_result = {
            "cycle": cycle_num,
            "type": "feedback",
            "flight_log": flight_log_path,
            "reward_path": new_reward_path,
            "model_path": os.path.join(model_dir, "student.pt"),
            "training_metrics": metrics,
            "flight_metrics": {
                "mean_position_error": report.overall_mean_position_error,
                "max_position_error": report.overall_max_position_error,
                "success_rate": report.overall_success_rate,
                "fallback_activated": report.fallback_activated,
                "anomaly_count": len(report.anomalies),
            },
            "sim_real_analysis": sim_real_results.get("iterations", [{}])[0].get("analysis", ""),
            "improved": improved,
            "timestamp": datetime.now().isoformat(),
        }

        self.state["current_cycle"] = cycle_num
        self.state["cycles"].append(cycle_result)
        self._save_state()

        logger.info(f"Cycle {cycle_num} complete. "
                     f"Error: {current_error:.3f}m, "
                     f"Improved: {improved}, "
                     f"Best: cycle {self.state['best_cycle']}")

        return cycle_result

    def export_best_model(self, output_dir: str = None) -> str:
        """Export the best model from the feedback loop to PX4 format."""
        if self.state["best_cycle"] is None:
            logger.error("No completed cycles to export from")
            return ""

        best_cycle = self.state["cycles"][self.state["best_cycle"]]
        model_path = best_cycle["model_path"]

        if output_dir is None:
            output_dir = str(Path(self.config_path).parent / ".." / ".." /
                             "src" / "modules" / "mc_nn_control")

        logger.info(f"Exporting best model (cycle {self.state['best_cycle']}) to {output_dir}")

        from export_tflite import export_to_onnx, convert_onnx_to_tflite, generate_cc_arrays
        import torch

        model = create_model_from_config(self.config)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        intermediate = os.path.join(self.workspace_dir, "export")
        os.makedirs(intermediate, exist_ok=True)

        onnx_path = os.path.join(intermediate, "best_model.onnx")
        tflite_path = os.path.join(intermediate, "best_model.tflite")

        export_to_onnx(model, onnx_path, self.config["model"]["input_size"])
        convert_onnx_to_tflite(onnx_path, tflite_path,
                               self.config.get("export", {}).get("quantization", "none"))
        generate_cc_arrays(tflite_path, output_dir)

        logger.info(f"Exported to {output_dir}/control_net.cpp")
        return output_dir

    def print_summary(self):
        """Print a summary of all feedback loop cycles."""
        print("\n" + "=" * 70)
        print("Runtime Distillation Feedback Loop Summary")
        print("=" * 70)
        print(f"Total cycles: {len(self.state['cycles'])}")
        print(f"Converged: {self.state['converged']}")
        print(f"Best cycle: {self.state['best_cycle']}")
        print(f"Best position error: {self.state['best_position_error']:.3f}m")
        print()

        for cycle in self.state["cycles"]:
            cycle_num = cycle["cycle"]
            cycle_type = cycle["type"]
            flight_metrics = cycle.get("flight_metrics", {})
            improved = cycle.get("improved", "N/A")

            if cycle_type == "initial":
                print(f"  Cycle {cycle_num} [INITIAL]: Model trained from LLM reward")
            else:
                error = flight_metrics.get("mean_position_error", "N/A")
                success = flight_metrics.get("success_rate", "N/A")
                anomalies = flight_metrics.get("anomaly_count", 0)
                marker = " *BEST*" if cycle_num == self.state["best_cycle"] else ""
                print(f"  Cycle {cycle_num} [FEEDBACK]: "
                      f"error={error:.3f}m, success={success:.2%}, "
                      f"anomalies={anomalies}, improved={improved}{marker}")

            # Show analysis excerpt if available
            analysis = cycle.get("sim_real_analysis", "")
            if analysis:
                first_line = analysis.strip().split("\n")[0][:80]
                print(f"    Analysis: {first_line}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop runtime distillation feedback orchestrator"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Run initial training (Eureka-style)")
    init_parser.add_argument("--config", type=str,
                             default=str(Path(__file__).parent / "model_config.yaml"))
    init_parser.add_argument("--scenarios", type=str,
                             default=str(Path(__file__).parent / "scenarios.yaml"))
    init_parser.add_argument("--scenario", type=str, default="hover_stable")
    init_parser.add_argument("--workspace", type=str, default=None)

    # Feedback command
    fb_parser = subparsers.add_parser("feedback", help="Run one feedback cycle from flight data")
    fb_parser.add_argument("--config", type=str,
                           default=str(Path(__file__).parent / "model_config.yaml"))
    fb_parser.add_argument("--scenarios", type=str,
                           default=str(Path(__file__).parent / "scenarios.yaml"))
    fb_parser.add_argument("--flight-log", type=str, required=True,
                           help="Flight log file (.ulg, .json, .csv)")
    fb_parser.add_argument("--sim-baseline", type=str, default=None)
    fb_parser.add_argument("--workspace", type=str, default=None)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export best model to PX4")
    export_parser.add_argument("--config", type=str,
                               default=str(Path(__file__).parent / "model_config.yaml"))
    export_parser.add_argument("--scenarios", type=str,
                               default=str(Path(__file__).parent / "scenarios.yaml"))
    export_parser.add_argument("--output", type=str, default=None)
    export_parser.add_argument("--workspace", type=str, default=None)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show feedback loop status")
    status_parser.add_argument("--config", type=str,
                               default=str(Path(__file__).parent / "model_config.yaml"))
    status_parser.add_argument("--scenarios", type=str,
                               default=str(Path(__file__).parent / "scenarios.yaml"))
    status_parser.add_argument("--workspace", type=str, default=None)

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return

    loop = DistillationFeedbackLoop(
        config_path=args.config,
        scenarios_path=args.scenarios,
        workspace_dir=args.workspace,
    )

    if args.command == "init":
        result = loop.run_initial_training(args.scenario)
        print(f"\nInitial training complete. Model: {result['model_path']}")
        print("Next: Deploy to PX4, fly, and run 'feedback' with the flight log.")

    elif args.command == "feedback":
        result = loop.run_feedback_cycle(
            flight_log_path=args.flight_log,
            sim_baseline_path=args.sim_baseline,
        )
        print(f"\nFeedback cycle {result['cycle']} complete.")
        if result.get("improved"):
            print("Performance IMPROVED from previous cycle.")
        loop.print_summary()

    elif args.command == "export":
        output = loop.export_best_model(args.output)
        print(f"\nExported best model to: {output}")

    elif args.command == "status":
        loop.print_summary()


if __name__ == "__main__":
    main()
