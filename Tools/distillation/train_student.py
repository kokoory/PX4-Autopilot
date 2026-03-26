#!/usr/bin/env python3
"""
Student MLP Training for Knowledge Distillation.

Trains a compact MLP that can run within PX4's mc_nn_control module
at 400Hz (2.5ms budget). Supports two training modes:

1. Behavior Cloning: Learn from expert demonstration data
2. RL Distillation: Train with LLM-generated reward functions

The MLP architecture is constrained to operations supported by
PX4's TFLite Micro OpResolver: FullyConnected, ReLU, Add.

Usage:
    python train_student.py --config model_config.yaml --mode behavior_cloning --data expert_data.npz
    python train_student.py --config model_config.yaml --mode rl_distillation --reward output/reward_hover_stable.py
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block using only FullyConnected + ReLU + Add ops."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        # Only use skip connection if dimensions match
        self.use_skip = (in_features == out_features)

    def forward(self, x):
        out = self.relu(self.linear(x))
        if self.use_skip:
            out = out + x  # Add op (TFLite Micro compatible)
        return out


class StudentMLP(nn.Module):
    """
    Compact MLP for real-time flight control.

    Architecture matches PX4 mc_nn_control requirements:
    - Input: 15 elements (pos_error, attitude_6d, lin_vel, ang_vel)
    - Output: 4 elements (motor commands)
    - Only uses FullyConnected, ReLU, Add operations
    """

    def __init__(
        self,
        input_size: int = 15,
        output_size: int = 4,
        hidden_layers: list = None,
        use_skip_connections: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 32]

        self.use_skip = use_skip_connections
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            if use_skip_connections and prev_size == hidden_size:
                layers.append(ResidualBlock(prev_size, hidden_size))
            else:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = self.features(x)
        return self.output_layer(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(config: dict) -> StudentMLP:
    """Create a StudentMLP from configuration dictionary."""
    model_config = config["model"]
    return StudentMLP(
        input_size=model_config["input_size"],
        output_size=model_config["output_size"],
        hidden_layers=model_config["hidden_layers"],
        use_skip_connections=model_config.get("use_skip_connections", True),
    )


def generate_synthetic_data(
    num_samples: int = 10000, seed: int = 42
) -> tuple:
    """
    Generate synthetic training data for testing the pipeline.
    In production, this data comes from SITL simulation with expert controllers.
    """
    rng = np.random.RandomState(seed)

    observations = np.zeros((num_samples, 15), dtype=np.float32)
    actions = np.zeros((num_samples, 4), dtype=np.float32)

    for i in range(num_samples):
        # Random position error
        pos_error = rng.uniform(-3.0, 3.0, 3)
        observations[i, 0:3] = pos_error

        # Random attitude (near identity rotation)
        angle = rng.uniform(-0.3, 0.3)
        observations[i, 3] = np.cos(angle)  # R[0,0]
        observations[i, 4] = -np.sin(angle)  # R[0,1]
        observations[i, 5] = 0.0  # R[0,2]
        observations[i, 6] = np.sin(angle)  # R[1,0]
        observations[i, 7] = np.cos(angle)  # R[1,1]
        observations[i, 8] = 0.0  # R[1,2]

        # Random velocities
        observations[i, 9:12] = rng.uniform(-2.0, 2.0, 3)
        observations[i, 12:15] = rng.uniform(-1.0, 1.0, 3)

        # Simple PD controller as "expert" for synthetic data
        kp = 0.5
        kd = 0.3
        thrust_base = 0.5  # Hover thrust

        # Simplified: motor commands from position error + velocity damping
        fx = -kp * pos_error[0] - kd * observations[i, 9]
        fy = -kp * pos_error[1] - kd * observations[i, 10]
        fz = -kp * pos_error[2] - kd * observations[i, 11] + thrust_base

        # Mix to 4 motors (simplified X-configuration)
        actions[i, 0] = np.clip(fz + fx + fy, -1.0, 1.0)
        actions[i, 1] = np.clip(fz - fx + fy, -1.0, 1.0)
        actions[i, 2] = np.clip(fz + fx - fy, -1.0, 1.0)
        actions[i, 3] = np.clip(fz - fx - fy, -1.0, 1.0)

    return observations, actions


def train_behavior_cloning(
    model: StudentMLP,
    config: dict,
    data_path: Optional[str] = None,
    output_dir: str = "output",
) -> dict:
    """Train the student MLP using behavior cloning from expert data."""
    train_config = config["training"]

    # Load or generate data
    if data_path and os.path.exists(data_path):
        data = np.load(data_path)
        observations = data["observations"]
        actions = data["actions"]
        logger.info(f"Loaded {len(observations)} samples from {data_path}")
    else:
        logger.info("No data file found, generating synthetic training data")
        observations, actions = generate_synthetic_data()

    # Split data
    n = len(observations)
    train_split = train_config.get("train_split", 0.8)
    val_split = train_config.get("validation_split", 0.1)

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create datasets
    train_obs = torch.FloatTensor(observations[train_idx])
    train_act = torch.FloatTensor(actions[train_idx])
    val_obs = torch.FloatTensor(observations[val_idx])
    val_act = torch.FloatTensor(actions[val_idx])
    test_obs = torch.FloatTensor(observations[test_idx])
    test_act = torch.FloatTensor(actions[test_idx])

    train_loader = DataLoader(
        TensorDataset(train_obs, train_act),
        batch_size=train_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_obs, val_act),
        batch_size=train_config["batch_size"],
    )

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.0001),
    )

    scheduler_config = train_config.get("scheduler", {})
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_config.get("T_max", train_config["epochs"]),
        eta_min=scheduler_config.get("eta_min", 1e-5),
    )

    criterion = nn.MSELoss()

    # Early stopping
    early_stop_config = train_config.get("early_stopping", {})
    patience = early_stop_config.get("patience", 20)
    min_delta = early_stop_config.get("min_delta", 0.0001)
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(output_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    # Training loop
    epochs = train_config["epochs"]
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)

            optimizer.zero_grad()
            pred = model(batch_obs)
            loss = criterion(pred, batch_act)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_obs)

        train_loss /= len(train_obs)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
                pred = model(batch_obs)
                loss = criterion(pred, batch_act)
                val_loss += loss.item() * len(batch_obs)
        val_loss /= len(val_obs)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:>4d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
    model.eval()

    with torch.no_grad():
        test_pred = model(test_obs.to(device))
        test_loss = criterion(test_pred, test_act.to(device)).item()
        test_pred_np = test_pred.cpu().numpy()
        test_act_np = test_act.numpy()
        max_error = np.max(np.abs(test_pred_np - test_act_np))
        mean_error = np.mean(np.abs(test_pred_np - test_act_np))

    # Save final model
    model_path = os.path.join(output_dir, "student.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {
        "test_loss": test_loss,
        "max_absolute_error": float(max_error),
        "mean_absolute_error": float(mean_error),
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "num_parameters": model.count_parameters(),
        "training_mode": "behavior_cloning",
    }

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete. Model saved to {model_path}")
    logger.info(f"Parameters: {model.count_parameters()}")
    logger.info(f"Test loss: {test_loss:.6f}, Max error: {max_error:.6f}")

    return metrics


def train_rl_distillation(
    model: StudentMLP,
    config: dict,
    reward_path: str,
    output_dir: str = "output",
) -> dict:
    """
    Train the student MLP using RL with LLM-generated reward functions.
    This is a simplified version - production would use PPO with SITL.
    """
    rl_config = config.get("rl_distillation", {})

    # Load reward function
    with open(reward_path) as f:
        reward_code = f.read()

    namespace = {"np": np, "numpy": np}
    exec(reward_code, namespace)
    reward_fn = namespace["reward"]

    logger.info(f"Loaded reward function from {reward_path}")

    # Simple policy gradient training (production would use PPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs(output_dir, exist_ok=True)

    total_timesteps = rl_config.get("total_timesteps", 100000)
    n_steps = rl_config.get("n_steps", 2048)
    gamma = rl_config.get("gamma", 0.99)
    num_iterations = total_timesteps // n_steps

    best_return = float("-inf")

    for iteration in range(num_iterations):
        # Collect trajectories
        observations = []
        actions = []
        rewards = []
        log_probs = []

        obs = np.zeros(15, dtype=np.float32)
        obs[0:3] = np.random.uniform(-2.0, 2.0, 3)
        obs[3] = 1.0
        obs[7] = 1.0

        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action_mean = model(obs_tensor)

            # Add exploration noise
            action_dist = torch.distributions.Normal(action_mean, 0.1)
            action = action_dist.sample()
            lp = action_dist.log_prob(action).sum(-1)

            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, -1.0, 1.0)

            # Simple dynamics
            next_obs = obs.copy()
            next_obs[0:3] += obs[9:12] * 0.01  # Position integration
            next_obs[9:12] = action_np[:3] * 0.5  # Velocity from actions
            next_obs[0:3] *= 0.99  # Slight damping

            r = reward_fn(obs, action_np, next_obs)

            observations.append(obs.copy())
            actions.append(action_np.copy())
            rewards.append(r)
            log_probs.append(lp.item())

            obs = next_obs

            # Reset if diverged
            if np.linalg.norm(obs[0:3]) > 10.0:
                obs = np.zeros(15, dtype=np.float32)
                obs[0:3] = np.random.uniform(-2.0, 2.0, 3)
                obs[3] = 1.0
                obs[7] = 1.0

        # Compute returns
        returns = np.zeros(n_steps)
        running_return = 0.0
        for t in reversed(range(n_steps)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        obs_batch = torch.FloatTensor(np.array(observations)).to(device)
        returns_batch = torch.FloatTensor(returns).to(device)
        log_probs_batch = torch.FloatTensor(np.array(log_probs)).to(device)

        loss = -(log_probs_batch * returns_batch).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        mean_return = np.mean(rewards) * n_steps
        if iteration % 10 == 0:
            logger.info(
                f"Iteration {iteration}/{num_iterations} | "
                f"Mean return: {mean_return:.2f} | Loss: {loss.item():.4f}"
            )

        if mean_return > best_return:
            best_return = mean_return
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Load best and save
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
    model_path = os.path.join(output_dir, "student.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {
        "best_return": float(best_return),
        "num_parameters": model.count_parameters(),
        "total_iterations": num_iterations,
        "training_mode": "rl_distillation",
    }

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"RL training complete. Best return: {best_return:.2f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train student MLP for flight control")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "model_config.yaml"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["behavior_cloning", "rl_distillation"],
        default="behavior_cloning",
    )
    parser.add_argument("--data", type=str, default=None, help="Expert data .npz file")
    parser.add_argument("--reward", type=str, default=None, help="Reward function .py file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = create_model_from_config(config)
    logger.info(f"Created model with {model.count_parameters()} parameters")
    logger.info(f"Architecture: {model}")

    output_dir = args.output_dir or str(Path(__file__).parent / "output")

    if args.mode == "behavior_cloning":
        metrics = train_behavior_cloning(model, config, args.data, output_dir)
    else:
        if not args.reward:
            logger.error("--reward is required for rl_distillation mode")
            return
        metrics = train_rl_distillation(model, config, args.reward, output_dir)

    print(f"\nTraining complete ({args.mode})")
    print(f"  Parameters: {metrics['num_parameters']}")
    for key, val in metrics.items():
        if key != "num_parameters":
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
