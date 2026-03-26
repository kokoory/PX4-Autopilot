#!/usr/bin/env python3
"""
Sim-to-Real Bridge: LLM-Guided Reward Refinement from Real Flight Data.

This is the core novel contribution of the Runtime Distillation Feedback Loop.
Unlike Eureka (which only uses simulation metrics), this module feeds REAL
flight performance data to an LLM, which then reasons about the sim-to-real
gap and generates improved reward functions that compensate for unmodeled
real-world effects.

Key insight: The LLM acts as a "physics reasoning engine" that hypothesizes
WHY real-world performance differs from simulation, and encodes those
hypotheses as reward function modifications.

Sim-to-real gap sources the LLM can reason about:
  - Motor response delay and nonlinearity
  - Aerodynamic ground effect and prop wash
  - Sensor noise and latency
  - Battery voltage sag under load
  - Vibration-induced estimation errors
  - Wind gusts and turbulence not in simulation
  - Center of gravity offset
  - Frame flex and structural compliance

Usage:
    python sim_real_bridge.py --flight-report analysis.json --current-reward reward.py
    python sim_real_bridge.py --flight-report analysis.json --current-reward reward.py --iterations 3
"""

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from flight_analyzer import (
    FlightAnalysisReport,
    format_report_for_llm,
    analyze_flight,
)
from reward_generator import validate_reward_function, load_reward_function

logger = logging.getLogger(__name__)

SIM_REAL_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert in UAV flight dynamics, sim-to-real transfer, and reinforcement
learning reward design. You have deep knowledge of:

1. Quadrotor physics: motor dynamics, aerodynamics, gyroscopic effects
2. Common sim-to-real gaps: motor lag, prop wash, ground effect, sensor noise
3. Control theory: PID tuning, frequency response, stability margins
4. Reward engineering: shaping rewards to produce robust, transferable policies

Your task: Given real-world flight performance data from a distilled neural
network controller, identify the likely causes of performance degradation
compared to simulation, and generate an improved reward function that will
produce a more robust controller when retrained.

CRITICAL RULES:
1. Function signature: def reward(obs, action, next_obs) -> float
2. Must import numpy inside the function if needed
3. Return a single float value
4. You must explain your reasoning about the sim-to-real gap BEFORE the code
5. The reward must specifically address the identified real-world issues
6. Output format: First a "## Analysis" section, then "## Reward Function" with ONLY the Python code

Observation space (15 elements):
- [0:3] Position error (x, y, z) in body frame, meters
- [3:9] Attitude as 6D rotation (first two columns of rotation matrix)
- [9:12] Linear velocity in body frame, m/s
- [12:15] Angular velocity in body frame, rad/s

Action space (4 elements):
- Motor commands in [-1, 1] range
""")


def build_sim_real_prompt(
    flight_report: str,
    current_reward_code: str,
    vehicle_params: dict = None,
    iteration: int = 1,
    previous_analysis: str = None,
) -> str:
    """
    Build the prompt that asks the LLM to analyze sim-to-real gap
    and improve the reward function.
    """
    sections = [
        f"# Sim-to-Real Reward Refinement (Iteration {iteration})",
        "",
        flight_report,
        "",
    ]

    if vehicle_params:
        sections.extend([
            "## Vehicle Parameters",
            f"- Mass: {vehicle_params.get('mass_kg', 'unknown')} kg",
            f"- Arm length: {vehicle_params.get('arm_length_m', 'unknown')} m",
            f"- Max RPM: {vehicle_params.get('max_rpm', 'unknown')}",
            f"- Motor count: {vehicle_params.get('num_motors', 4)}",
            "",
        ])

    sections.extend([
        "## Current Reward Function",
        "```python",
        current_reward_code,
        "```",
        "",
    ])

    if previous_analysis:
        sections.extend([
            "## Previous Iteration Analysis",
            previous_analysis,
            "",
        ])

    sections.extend([
        "## Your Task",
        "",
        "1. **Analyze the sim-to-real gap**: Based on the flight data above, identify",
        "   the most likely physical causes of performance degradation. Consider:",
        "   - Are certain axes worse than others? (asymmetric dynamics)",
        "   - Is motor saturation indicating insufficient thrust margin?",
        "   - Are oscillations present? (unmodeled delays or vibrations)",
        "   - Does performance degrade over time? (thermal or battery effects)",
        "   - Are specific flight phases (hover vs. maneuver) disproportionately affected?",
        "",
        "2. **Design reward modifications**: For each identified cause, explain what",
        "   reward term will help the RL agent learn to compensate. Be specific about",
        "   the physical reasoning.",
        "",
        "3. **Generate improved reward function**: Output the complete Python function",
        "   that incorporates your modifications. The function must be self-contained.",
        "",
        "Remember: The goal is NOT to tune PID gains, but to shape the reward so that",
        "the neural network policy learns to be robust to real-world effects that are",
        "absent from the simulation.",
    ])

    return "\n".join(sections)


def call_llm_for_sim_real(prompt: str) -> Optional[dict]:
    """
    Call LLM for sim-to-real analysis. Returns both the analysis text
    and the reward function code.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SIM_REAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text

        # Parse response: extract analysis and reward function
        analysis = ""
        reward_code = ""

        if "## Analysis" in response_text:
            parts = response_text.split("## Reward Function")
            analysis = parts[0].replace("## Analysis", "").strip()

            if len(parts) > 1:
                code_part = parts[1]
                # Extract code block
                if "```python" in code_part:
                    code_start = code_part.index("```python") + len("```python")
                    code_end = code_part.index("```", code_start)
                    reward_code = code_part[code_start:code_end].strip()
                elif "def reward" in code_part:
                    # Find the function definition
                    func_start = code_part.index("def reward")
                    reward_code = code_part[func_start:].strip()
        elif "def reward" in response_text:
            # Fallback: try to extract just the code
            if "```python" in response_text:
                code_start = response_text.index("```python") + len("```python")
                code_end = response_text.index("```", code_start)
                reward_code = response_text[code_start:code_end].strip()
            else:
                func_start = response_text.index("def reward")
                reward_code = response_text[func_start:].strip()

            analysis = response_text[:response_text.index("def reward")].strip()

        return {
            "analysis": analysis,
            "reward_code": reward_code,
            "full_response": response_text,
        }

    except ImportError:
        logger.warning("anthropic package not installed")
        return None
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None


def generate_sim_real_reward(
    flight_report_path: str,
    current_reward_path: str,
    vehicle_params: dict = None,
    sim_baseline_path: str = None,
    output_dir: str = None,
    iterations: int = 1,
) -> dict:
    """
    Main pipeline: analyze real flight data → LLM reasons about gap → improved reward.
    """
    # Load flight report
    if flight_report_path.endswith(".json"):
        with open(flight_report_path) as f:
            report_data = json.load(f)

        # Reconstruct report object for formatting
        report = FlightAnalysisReport(**{
            k: v for k, v in report_data.items()
            if k in FlightAnalysisReport.__dataclass_fields__
        })
        flight_report_text = format_report_for_llm(report)
    else:
        # Analyze raw log directly
        sim_baseline = None
        if sim_baseline_path:
            with open(sim_baseline_path) as f:
                sim_baseline = json.load(f)

        report = analyze_flight(flight_report_path, sim_baseline)
        flight_report_text = format_report_for_llm(report)

    # Load current reward
    with open(current_reward_path) as f:
        current_reward_code = f.read()

    if output_dir is None:
        output_dir = str(Path(flight_report_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "iterations": [],
        "initial_reward_path": current_reward_path,
        "flight_report_path": flight_report_path,
    }

    previous_analysis = None
    best_reward_code = current_reward_code

    for i in range(1, iterations + 1):
        logger.info(f"=== Sim-to-Real iteration {i}/{iterations} ===")

        # Build prompt
        prompt = build_sim_real_prompt(
            flight_report=flight_report_text,
            current_reward_code=current_reward_code,
            vehicle_params=vehicle_params,
            iteration=i,
            previous_analysis=previous_analysis,
        )

        # Call LLM
        llm_result = call_llm_for_sim_real(prompt)

        if llm_result is None:
            logger.warning(f"Iteration {i}: LLM call failed, using fallback")
            llm_result = _generate_fallback_reward(report, current_reward_code)

        # Validate the generated reward
        if llm_result["reward_code"] and validate_reward_function(llm_result["reward_code"]):
            logger.info(f"Iteration {i}: Valid reward function generated")
            best_reward_code = llm_result["reward_code"]
            current_reward_code = best_reward_code
        else:
            logger.warning(f"Iteration {i}: Invalid reward, keeping previous")

        # Save iteration results
        iter_result = {
            "iteration": i,
            "analysis": llm_result.get("analysis", ""),
            "reward_valid": validate_reward_function(llm_result.get("reward_code", "")),
        }
        results["iterations"].append(iter_result)

        # Save reward function
        reward_path = os.path.join(output_dir, f"reward_sim_real_v{i}.py")
        with open(reward_path, "w") as f:
            f.write(best_reward_code)

        # Save analysis
        analysis_path = os.path.join(output_dir, f"analysis_v{i}.md")
        with open(analysis_path, "w") as f:
            f.write(llm_result.get("analysis", ""))

        previous_analysis = llm_result.get("analysis", "")
        logger.info(f"Saved: {reward_path}")

    # Save final results
    results["final_reward_path"] = os.path.join(output_dir, f"reward_sim_real_v{iterations}.py")
    results_path = os.path.join(output_dir, "sim_real_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _generate_fallback_reward(report: FlightAnalysisReport, current_code: str) -> dict:
    """
    Generate an improved reward function without LLM, based on heuristic
    analysis of the flight report. Used as fallback when API is unavailable.
    """
    gap = report.sim_real_gap_indicators if isinstance(report, FlightAnalysisReport) else {}

    # Analyze the gap indicators to determine which terms to adjust
    adjustments = []

    steady_error = gap.get("steady_state_error", 0)
    if steady_error > 0.5:
        adjustments.append("# Increased position penalty due to high steady-state error")
        pos_weight = min(0.6, 0.4 + steady_error * 0.1)
    else:
        pos_weight = 0.40

    vibration = gap.get("vibration_magnitude", 0)
    if vibration > 0.5:
        adjustments.append("# Added vibration damping term due to high angular velocity noise")
        ang_vel_weight = min(0.25, 0.10 + vibration * 0.1)
    else:
        ang_vel_weight = 0.10

    saturation_freq = gap.get("saturation_frequency", 0)
    if saturation_freq > 0.1:
        adjustments.append("# Increased action smoothness penalty due to motor saturation")
        action_weight = min(0.20, 0.10 + saturation_freq * 0.2)
    else:
        action_weight = 0.10

    motor_asymmetry = gap.get("motor_asymmetry", 0)
    if motor_asymmetry > 0.05:
        adjustments.append("# Added motor balance term due to detected asymmetry")

    adjustments_comment = "\n    ".join(adjustments) if adjustments else "# No significant sim-to-real gaps detected"

    reward_code = textwrap.dedent(f"""\
def reward(obs: 'np.ndarray', action: 'np.ndarray', next_obs: 'np.ndarray') -> float:
    \"\"\"
    Sim-to-real refined reward function.
    Auto-generated from flight data analysis (fallback mode).

    Adjustments based on real-world performance:
    {adjustments_comment}
    \"\"\"
    import numpy as np

    # Position error (primary objective, weight adjusted from real-world data)
    pos_error = np.linalg.norm(next_obs[0:3])
    pos_reward = np.exp(-2.5 * pos_error)

    # Attitude stability
    att_error = np.sqrt((next_obs[3] - 1.0)**2 + next_obs[4]**2 + next_obs[5]**2 +
                        next_obs[6]**2 + (next_obs[7] - 1.0)**2 + next_obs[8]**2)
    att_reward = np.exp(-1.5 * att_error)

    # Velocity damping
    vel_magnitude = np.linalg.norm(next_obs[9:12])
    vel_reward = np.exp(-0.5 * vel_magnitude)

    # Angular velocity damping (increased if vibrations detected)
    ang_vel = np.linalg.norm(next_obs[12:15])
    ang_vel_reward = np.exp(-0.5 * ang_vel)

    # Action smoothness (penalize large commands that may cause saturation)
    action_magnitude = np.linalg.norm(action)
    action_reward = np.exp(-0.2 * action_magnitude)

    # Motor balance: penalize asymmetric motor commands (real-world robustness)
    motor_variance = np.var(action)
    balance_reward = np.exp(-0.5 * motor_variance)

    # Action rate penalty: approximate smoothness from action magnitude
    # (In full RL pipeline, previous action would be tracked for actual rate)
    rate_penalty = np.exp(-0.1 * np.max(np.abs(action)))

    total = ({pos_weight:.2f} * pos_reward +
             0.20 * att_reward +
             0.10 * vel_reward +
             {ang_vel_weight:.2f} * ang_vel_reward +
             {action_weight:.2f} * action_reward +
             0.05 * balance_reward +
             0.05 * rate_penalty)

    return float(total)
""")

    analysis = textwrap.dedent(f"""\
Heuristic sim-to-real gap analysis (fallback mode, no LLM):
- Steady-state position error: {steady_error:.3f}m
- Vibration magnitude: {vibration:.3f} rad/s
- Motor saturation frequency: {saturation_freq:.1%}
- Motor asymmetry: {motor_asymmetry:.4f}

Reward adjustments applied:
- Position weight: {pos_weight:.2f} (default: 0.40)
- Angular velocity weight: {ang_vel_weight:.2f} (default: 0.10)
- Action smoothness weight: {action_weight:.2f} (default: 0.10)
""")

    return {
        "analysis": analysis,
        "reward_code": reward_code,
        "full_response": analysis + "\n\n" + reward_code,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided sim-to-real reward refinement from flight data"
    )
    parser.add_argument("--flight-report", type=str, required=True,
                        help="Flight analysis report (.json) or raw log (.ulg, .csv)")
    parser.add_argument("--current-reward", type=str, required=True,
                        help="Current reward function (.py)")
    parser.add_argument("--vehicle-params", type=str, default=None,
                        help="Vehicle parameters YAML (scenarios.yaml)")
    parser.add_argument("--sim-baseline", type=str, default=None,
                        help="Simulation baseline metrics JSON")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of LLM refinement iterations")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    vehicle_params = None
    if args.vehicle_params:
        with open(args.vehicle_params) as f:
            scenarios = yaml.safe_load(f)
            vehicle_params = scenarios.get("vehicle_defaults", {})

    results = generate_sim_real_reward(
        flight_report_path=args.flight_report,
        current_reward_path=args.current_reward,
        vehicle_params=vehicle_params,
        sim_baseline_path=args.sim_baseline,
        output_dir=args.output_dir,
        iterations=args.iterations,
    )

    print(f"\nSim-to-Real refinement complete ({len(results['iterations'])} iterations)")
    print(f"Final reward: {results['final_reward_path']}")
    for it in results["iterations"]:
        valid = "VALID" if it["reward_valid"] else "INVALID"
        print(f"  Iteration {it['iteration']}: [{valid}]")
        if it["analysis"]:
            # Print first 2 lines of analysis
            first_lines = it["analysis"].strip().split("\n")[:2]
            for line in first_lines:
                print(f"    {line}")


if __name__ == "__main__":
    main()
