#!/usr/bin/env python3
"""
Eureka-style Reward Function Generator for Flight Control Knowledge Distillation.

Uses LLM (Claude) to generate reward functions for RL-based policy training.
The LLM analyzes vehicle parameters and mission objectives to create reward
functions that guide the RL agent toward optimal flight control behavior.

Iterative improvement: simulation metrics are fed back to the LLM to refine
the reward function over multiple iterations.

Usage:
    python reward_generator.py --config model_config.yaml --scenarios scenarios.yaml
    python reward_generator.py --iterations 5 --scenario hover_stable
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default reward function used as seed / fallback
DEFAULT_REWARD_FUNCTION = textwrap.dedent("""\
def reward(obs: 'np.ndarray', action: 'np.ndarray', next_obs: 'np.ndarray') -> float:
    \"\"\"
    Reward function for quadrotor hover control.

    obs/next_obs: 15-element vector
        [0:3]  position error (x, y, z) in meters
        [3:9]  attitude 6D (first two columns of rotation matrix)
        [9:12] linear velocity (vx, vy, vz) in m/s
        [12:15] angular velocity (wx, wy, wz) in rad/s

    action: 4-element motor command vector in [-1, 1]
    \"\"\"
    import numpy as np

    # Position error penalty (most important)
    pos_error = np.linalg.norm(next_obs[0:3])
    pos_reward = np.exp(-2.0 * pos_error)

    # Attitude penalty: identity rotation means upright
    # For identity rotation matrix, columns should be [1,0,0] and [0,1,0]
    att_error = np.sqrt((next_obs[3] - 1.0)**2 + next_obs[4]**2 + next_obs[5]**2 +
                        next_obs[6]**2 + (next_obs[7] - 1.0)**2 + next_obs[8]**2)
    att_reward = np.exp(-1.0 * att_error)

    # Velocity penalty (prefer low velocity near target)
    vel_magnitude = np.linalg.norm(next_obs[9:12])
    vel_reward = np.exp(-0.5 * vel_magnitude)

    # Angular velocity penalty (prefer smooth rotation)
    ang_vel_magnitude = np.linalg.norm(next_obs[12:15])
    ang_vel_reward = np.exp(-0.3 * ang_vel_magnitude)

    # Action smoothness penalty
    action_magnitude = np.linalg.norm(action)
    action_reward = np.exp(-0.1 * action_magnitude)

    # Weighted combination
    total_reward = (0.40 * pos_reward +
                    0.25 * att_reward +
                    0.15 * vel_reward +
                    0.10 * ang_vel_reward +
                    0.10 * action_reward)

    return float(total_reward)
""")

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert in quadrotor flight control and reinforcement learning reward design.
Your task is to generate Python reward functions for training a neural network flight controller.

The controller takes a 15-element observation vector and outputs 4 motor commands.

Observation space (15 elements):
- [0:3] Position error (x, y, z) in body frame, meters
- [3:9] Attitude as 6D rotation (first two columns of rotation matrix)
- [9:12] Linear velocity in body frame, m/s
- [12:15] Angular velocity in body frame, rad/s

Action space (4 elements):
- Motor commands in [-1, 1] range

Rules for the reward function:
1. Function signature: def reward(obs, action, next_obs) -> float
2. Must import numpy inside the function if needed
3. Return a single float value
4. Higher values indicate better performance
5. Must penalize large position errors as the primary objective
6. Must penalize excessive angular velocities for flight stability
7. Should encourage energy efficiency
8. Must be numerically stable (no division by zero, handle edge cases)
9. Output ONLY the Python function, no explanation or markdown
""")


def build_improvement_prompt(
    scenario: dict,
    vehicle: dict,
    metrics: dict,
    previous_reward_code: str,
    iteration: int,
    real_world_context: str = None,
) -> str:
    """Build the prompt for the LLM to improve the reward function.

    Args:
        real_world_context: Optional flight analysis report from the runtime
            feedback loop. When provided, the LLM should consider sim-to-real
            gap indicators when refining the reward function.
    """
    real_world_section = ""
    if real_world_context:
        real_world_section = textwrap.dedent(f"""\

Real-World Flight Feedback (from runtime distillation feedback loop):
{real_world_context}

IMPORTANT: The above data is from ACTUAL FLIGHT, not simulation. Use it to
identify sim-to-real discrepancies and adjust the reward to produce a controller
that is robust to these real-world effects.
""")

    return textwrap.dedent(f"""\
Iteration {iteration}: Improve the reward function for the following scenario.

Vehicle Parameters:
- Mass: {vehicle['mass_kg']} kg
- Arm length: {vehicle['arm_length_m']} m
- Max RPM: {vehicle['max_rpm']}
- Number of motors: {vehicle['num_motors']}

Scenario: {scenario.get('description', 'Unknown')}
Wind: {scenario.get('wind', {}).get('speed_ms', 0)} m/s

Previous reward function performance metrics:
- Mean position error: {metrics.get('mean_position_error', 'N/A')} m
- Max position error: {metrics.get('max_position_error', 'N/A')} m
- Mean attitude error: {metrics.get('mean_attitude_error', 'N/A')} deg
- Mean angular velocity: {metrics.get('mean_angular_velocity', 'N/A')} rad/s
- Episode return: {metrics.get('episode_return', 'N/A')}
- Settling time: {metrics.get('settling_time', 'N/A')} s

Previous reward function:
```python
{previous_reward_code}
```

{real_world_section}Based on these metrics, generate an improved reward function. Focus on:
1. Reducing position error if it's above the target threshold
2. Improving stability if angular velocities are too high
3. Achieving faster settling time
4. Maintaining energy efficiency

Output ONLY the improved Python function, no explanation.
""")


def validate_reward_function(code: str) -> bool:
    """Validate that the generated reward function is callable and returns a float."""
    try:
        namespace = {"np": np, "numpy": np}
        exec(code, namespace)

        if "reward" not in namespace:
            logger.error("Generated code does not define a 'reward' function")
            return False

        reward_fn = namespace["reward"]

        # Test with dummy data
        obs = np.zeros(15, dtype=np.float32)
        obs[3] = 1.0  # Identity rotation col1 x
        obs[7] = 1.0  # Identity rotation col2 y
        action = np.zeros(4, dtype=np.float32)
        next_obs = obs.copy()

        result = reward_fn(obs, action, next_obs)

        if not isinstance(result, (int, float, np.floating)):
            logger.error(f"Reward function returned {type(result)}, expected float")
            return False

        if not np.isfinite(result):
            logger.error(f"Reward function returned non-finite value: {result}")
            return False

        return True

    except Exception as e:
        logger.error(f"Reward function validation failed: {e}")
        return False


def load_reward_function(code: str):
    """Load and return the reward function from code string."""
    namespace = {"np": np, "numpy": np}
    exec(code, namespace)
    return namespace["reward"]


def call_llm(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    """Call Claude API to generate a reward function."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except ImportError:
        logger.warning("anthropic package not installed, using default reward function")
        return None
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None


def generate_initial_reward(scenario: dict, vehicle: dict) -> str:
    """Generate an initial reward function using LLM."""
    prompt = textwrap.dedent(f"""\
Generate a reward function for training a quadrotor neural network controller.

Vehicle Parameters:
- Mass: {vehicle['mass_kg']} kg
- Arm length: {vehicle['arm_length_m']} m
- Max RPM: {vehicle['max_rpm']}
- Thrust coefficient: {vehicle['thrust_coefficient']}
- Number of motors: {vehicle['num_motors']}

Mission: {scenario.get('description', 'Stable hover')}
Wind conditions: {scenario.get('wind', {}).get('speed_ms', 0)} m/s at {scenario.get('wind', {}).get('direction_deg', 0)} degrees
Success criteria: {json.dumps(scenario.get('success_criteria', {}), indent=2)}

Generate the reward function now.
""")

    response = call_llm(prompt)

    if response and validate_reward_function(response):
        logger.info("LLM generated valid initial reward function")
        return response

    logger.info("Falling back to default reward function")
    return DEFAULT_REWARD_FUNCTION


def iterate_reward(
    scenario: dict,
    vehicle: dict,
    current_reward_code: str,
    metrics: dict,
    iteration: int,
) -> str:
    """Use LLM to improve the reward function based on training metrics."""
    prompt = build_improvement_prompt(
        scenario, vehicle, metrics, current_reward_code, iteration
    )
    response = call_llm(prompt)

    if response and validate_reward_function(response):
        logger.info(f"Iteration {iteration}: LLM generated improved reward function")
        return response

    logger.warning(f"Iteration {iteration}: LLM improvement failed, keeping previous")
    return current_reward_code


def simulate_metrics(reward_fn, scenario: dict, num_episodes: int = 10) -> dict:
    """
    Simulate training metrics using the reward function.
    In production, this would run actual SITL simulation with PX4.
    This is a simplified placeholder for the pipeline.
    """
    position_errors = []
    attitude_errors = []
    angular_velocities = []
    episode_returns = []

    for _ in range(num_episodes):
        episode_return = 0.0
        obs = np.zeros(15, dtype=np.float32)
        # Start with some position error
        obs[0:3] = np.random.uniform(-2.0, 2.0, 3)
        obs[3] = 1.0  # Identity rotation
        obs[7] = 1.0

        for step in range(200):
            action = np.random.uniform(-1.0, 1.0, 4).astype(np.float32)
            # Simple dynamics simulation
            next_obs = obs.copy()
            next_obs[0:3] *= 0.98  # Position error decays
            next_obs[9:12] = action[:3] * 0.1  # Velocity from actions
            next_obs[12:15] = np.random.uniform(-0.1, 0.1, 3)  # Small angular vel

            r = reward_fn(obs, action, next_obs)
            episode_return += r
            obs = next_obs

            position_errors.append(np.linalg.norm(obs[0:3]))
            angular_velocities.append(np.linalg.norm(obs[12:15]))

        episode_returns.append(episode_return)

    return {
        "mean_position_error": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "mean_attitude_error": 0.0,  # Placeholder
        "mean_angular_velocity": float(np.mean(angular_velocities)),
        "episode_return": float(np.mean(episode_returns)),
        "settling_time": 0.0,  # Placeholder
    }


def run_pipeline(
    config_path: str,
    scenarios_path: str,
    scenario_name: Optional[str] = None,
    iterations: int = 5,
    output_dir: Optional[str] = None,
) -> dict:
    """Run the full reward generation pipeline."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(scenarios_path) as f:
        scenarios = yaml.safe_load(f)

    vehicle = scenarios["vehicle_defaults"]

    if scenario_name:
        scenario_list = {scenario_name: scenarios["scenarios"][scenario_name]}
    else:
        scenario_list = scenarios["scenarios"]

    if output_dir is None:
        output_dir = str(Path(config_path).parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for name, scenario in scenario_list.items():
        logger.info(f"=== Generating reward for scenario: {name} ===")

        # Generate initial reward function
        reward_code = generate_initial_reward(scenario, vehicle)
        reward_fn = load_reward_function(reward_code)

        best_reward_code = reward_code
        best_metrics = simulate_metrics(reward_fn, scenario)
        best_error = best_metrics["mean_position_error"]

        logger.info(f"Initial metrics: {json.dumps(best_metrics, indent=2)}")

        # Iterative improvement
        max_iterations = config.get("rl_distillation", {}).get(
            "reward_function_iterations", iterations
        )

        for i in range(1, max_iterations + 1):
            metrics = simulate_metrics(reward_fn, scenario)
            reward_code = iterate_reward(
                scenario, vehicle, reward_code, metrics, i
            )
            reward_fn = load_reward_function(reward_code)

            current_error = metrics["mean_position_error"]
            if current_error < best_error:
                best_error = current_error
                best_reward_code = reward_code
                best_metrics = metrics
                logger.info(f"Iteration {i}: Improved! Error: {current_error:.4f}")
            else:
                logger.info(f"Iteration {i}: No improvement. Error: {current_error:.4f}")

        # Save best reward function
        reward_path = os.path.join(output_dir, f"reward_{name}.py")
        with open(reward_path, "w") as f:
            f.write(best_reward_code)

        metrics_path = os.path.join(output_dir, f"metrics_{name}.json")
        with open(metrics_path, "w") as f:
            json.dump(best_metrics, f, indent=2)

        results[name] = {
            "reward_path": reward_path,
            "metrics": best_metrics,
        }

        logger.info(f"Saved reward function to {reward_path}")
        logger.info(f"Best metrics: {json.dumps(best_metrics, indent=2)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Eureka-style reward function generator for flight control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "model_config.yaml"),
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(Path(__file__).parent / "scenarios.yaml"),
        help="Path to scenarios YAML",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Specific scenario name to generate reward for",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of reward improvement iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated reward functions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_pipeline(
        config_path=args.config,
        scenarios_path=args.scenarios,
        scenario_name=args.scenario,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )

    print(f"\nGenerated {len(results)} reward function(s)")
    for name, result in results.items():
        print(f"  {name}: {result['reward_path']}")
        print(f"    Mean position error: {result['metrics']['mean_position_error']:.4f} m")


if __name__ == "__main__":
    main()
