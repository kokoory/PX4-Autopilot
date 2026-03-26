#!/usr/bin/env python3
"""
Flight Data Analyzer for Runtime Distillation Feedback Loop.

Parses PX4 ULog flight logs to extract DistillationStatus and sensor data,
then computes structured performance reports that can be fed to an LLM
for sim-to-real gap analysis.

The analyzer identifies failure patterns, performance anomalies, and
flight-phase-specific degradation that the LLM uses to improve reward functions.

Usage:
    python flight_analyzer.py --log flight.ulg --output analysis.json
    python flight_analyzer.py --log flight.ulg --format llm-prompt
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Flight phase names matching DistillationStatus.msg constants
PHASE_NAMES = {
    0: "idle",
    1: "takeoff",
    2: "hover",
    3: "cruise",
    4: "maneuver",
    5: "landing",
}


@dataclass
class PhaseStatistics:
    """Performance statistics for a single flight phase."""
    phase_name: str
    duration_s: float = 0.0
    sample_count: int = 0
    mean_position_error: float = 0.0
    max_position_error: float = 0.0
    std_position_error: float = 0.0
    mean_inference_time_us: float = 0.0
    max_inference_time_us: float = 0.0
    mean_motor_saturation: float = 0.0
    max_motor_saturation: float = 0.0
    mean_output_variance: float = 0.0
    mean_output_rate_of_change: float = 0.0
    mean_angular_velocity_magnitude: float = 0.0
    max_angular_velocity_magnitude: float = 0.0
    fallback_count: int = 0
    failed_inference_count: int = 0
    # Per-axis position error breakdown
    mean_error_x: float = 0.0
    mean_error_y: float = 0.0
    mean_error_z: float = 0.0
    # Motor output statistics
    motor_mean: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    motor_std: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class AnomalyEvent:
    """A detected performance anomaly during flight."""
    timestamp_s: float
    flight_phase: str
    anomaly_type: str  # "position_spike", "motor_saturation", "inference_timeout", "oscillation"
    severity: str  # "low", "medium", "high"
    description: str
    context: dict = field(default_factory=dict)


@dataclass
class FlightAnalysisReport:
    """Complete flight analysis report for LLM feedback."""
    log_file: str
    model_id: int = 0
    model_version: int = 0
    total_flight_time_s: float = 0.0
    total_inferences: int = 0
    overall_success_rate: float = 1.0
    overall_mean_position_error: float = 0.0
    overall_max_position_error: float = 0.0
    fallback_activated: bool = False
    phase_statistics: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    sim_real_gap_indicators: dict = field(default_factory=dict)


def parse_ulog(log_path: str) -> dict:
    """
    Parse a PX4 ULog file and extract relevant topics.
    Uses pyulog if available, otherwise falls back to CSV-based parsing.
    """
    try:
        from pyulog import ULog
        ulog = ULog(log_path)

        data = {}
        topic_map = {
            "distillation_status": "distillation_status",
            "neural_control": "neural_control",
            "vehicle_local_position": "vehicle_local_position",
            "vehicle_attitude": "vehicle_attitude",
            "vehicle_angular_velocity": "vehicle_angular_velocity",
            "sensor_combined": "sensor_combined",
            "actuator_motors": "actuator_motors",
        }

        for topic_name, key in topic_map.items():
            for d in ulog.data_list:
                if d.name == topic_name:
                    data[key] = {field: np.array(d.data[field]) for field in d.data}
                    break

        logger.info(f"Parsed ULog: {list(data.keys())}")
        return data

    except ImportError:
        logger.warning("pyulog not installed, attempting JSON fallback")
        return _parse_json_log(log_path)

    except Exception as e:
        logger.error(f"Failed to parse ULog: {e}")
        return {}


def _parse_json_log(log_path: str) -> dict:
    """Parse a JSON-format flight log (for testing/simulation)."""
    try:
        with open(log_path) as f:
            return json.load(f)
    except Exception:
        return {}


def parse_distillation_log(log_path: str) -> dict:
    """
    Parse a simplified distillation status log (CSV or JSON).
    Useful for SITL testing where full ULog may not be available.
    """
    path = Path(log_path)

    if path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            # List of status records
            return {"distillation_status": {
                "timestamp": np.array([r.get("timestamp", i) for i, r in enumerate(raw)]),
                "model_id": np.array([r.get("model_id", 0) for r in raw]),
                "inference_time_us": np.array([r.get("inference_time_us", 0) for r in raw]),
                "mean_position_error": np.array([r.get("mean_position_error", 0) for r in raw]),
                "motor_saturation_ratio": np.array([r.get("motor_saturation_ratio", 0) for r in raw]),
                "output_variance": np.array([r.get("output_variance", 0) for r in raw]),
                "output_rate_of_change": np.array([r.get("output_rate_of_change", 0) for r in raw]),
                "flight_phase": np.array([r.get("flight_phase", 0) for r in raw]),
                "fallback_active": np.array([r.get("fallback_active", False) for r in raw]),
                "success_rate": np.array([r.get("success_rate", 1.0) for r in raw]),
                "position_error_ned": np.array([r.get("position_error_ned", [0, 0, 0]) for r in raw]),
                "angular_velocity_raw": np.array([r.get("angular_velocity_raw", [0, 0, 0]) for r in raw]),
                "motor_outputs": np.array([r.get("motor_outputs", [0, 0, 0, 0]) for r in raw]),
            }}

    elif path.suffix == ".csv":
        data = np.genfromtxt(path, delimiter=",", names=True)
        return {"distillation_status": {name: data[name] for name in data.dtype.names}}

    return {}


def detect_anomalies(data: dict, config: dict = None) -> list:
    """Detect performance anomalies in flight data."""
    if config is None:
        config = {
            "position_error_spike_threshold": 1.5,  # meters
            "motor_saturation_threshold": 0.8,
            "inference_timeout_us": 2500,
            "oscillation_frequency_threshold": 5.0,  # Hz
        }

    anomalies = []
    ds = data.get("distillation_status", {})

    if not ds:
        return anomalies

    timestamps = ds.get("timestamp", np.array([]))
    if len(timestamps) == 0:
        return anomalies

    # Convert timestamps to seconds from start
    t0 = timestamps[0]
    time_s = (timestamps - t0) * 1e-6 if timestamps[0] > 1e10 else timestamps.astype(float)

    pos_errors = ds.get("mean_position_error", np.array([]))
    saturation = ds.get("motor_saturation_ratio", np.array([]))
    inf_times = ds.get("inference_time_us", np.array([]))
    phases = ds.get("flight_phase", np.array([]))

    # 1. Position error spikes
    if len(pos_errors) > 0:
        threshold = config["position_error_spike_threshold"]
        spike_mask = pos_errors > threshold

        for i in np.where(spike_mask)[0]:
            phase = PHASE_NAMES.get(int(phases[i]) if len(phases) > i else 0, "unknown")
            anomalies.append(AnomalyEvent(
                timestamp_s=float(time_s[i]),
                flight_phase=phase,
                anomaly_type="position_spike",
                severity="high" if pos_errors[i] > threshold * 2 else "medium",
                description=f"Position error {pos_errors[i]:.2f}m exceeds threshold {threshold}m",
                context={
                    "position_error": float(pos_errors[i]),
                    "saturation": float(saturation[i]) if len(saturation) > i else 0,
                },
            ))

    # 2. Motor saturation events
    if len(saturation) > 0:
        threshold = config["motor_saturation_threshold"]
        sat_mask = saturation > threshold

        # Group consecutive saturation into events
        changes = np.diff(sat_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        for start_idx in starts:
            phase = PHASE_NAMES.get(int(phases[start_idx]) if len(phases) > start_idx else 0, "unknown")
            anomalies.append(AnomalyEvent(
                timestamp_s=float(time_s[start_idx]),
                flight_phase=phase,
                anomaly_type="motor_saturation",
                severity="medium",
                description=f"Motor saturation {saturation[start_idx]:.1%} during {phase}",
                context={
                    "saturation_ratio": float(saturation[start_idx]),
                    "position_error": float(pos_errors[start_idx]) if len(pos_errors) > start_idx else 0,
                },
            ))

    # 3. Inference timeouts
    if len(inf_times) > 0:
        timeout = config["inference_timeout_us"]
        timeout_mask = inf_times > timeout

        for i in np.where(timeout_mask)[0]:
            anomalies.append(AnomalyEvent(
                timestamp_s=float(time_s[i]),
                flight_phase=PHASE_NAMES.get(int(phases[i]) if len(phases) > i else 0, "unknown"),
                anomaly_type="inference_timeout",
                severity="high",
                description=f"Inference time {inf_times[i]}us exceeds budget {timeout}us",
                context={"inference_time_us": int(inf_times[i])},
            ))

    # 4. Output oscillation detection (high-frequency motor command changes)
    output_roc = ds.get("output_rate_of_change", np.array([]))
    if len(output_roc) > 10:
        # Check for sustained high rate of change
        window = 10
        for i in range(window, len(output_roc)):
            window_mean = np.mean(output_roc[i - window:i])
            if window_mean > 0.3:  # Significant oscillation threshold
                phase = PHASE_NAMES.get(int(phases[i]) if len(phases) > i else 0, "unknown")
                anomalies.append(AnomalyEvent(
                    timestamp_s=float(time_s[i]),
                    flight_phase=phase,
                    anomaly_type="oscillation",
                    severity="medium",
                    description=f"Motor output oscillation detected (rate: {window_mean:.3f})",
                    context={
                        "rate_of_change": float(window_mean),
                        "position_error": float(pos_errors[i]) if len(pos_errors) > i else 0,
                    },
                ))
                break  # Report once per window

    return anomalies


def compute_phase_statistics(data: dict) -> list:
    """Compute per-flight-phase performance statistics."""
    ds = data.get("distillation_status", {})
    if not ds:
        return []

    phases_data = ds.get("flight_phase", np.array([]))
    if len(phases_data) == 0:
        return []

    timestamps = ds.get("timestamp", np.array([]))
    unique_phases = np.unique(phases_data)
    stats_list = []

    for phase_id in unique_phases:
        mask = phases_data == phase_id
        phase_name = PHASE_NAMES.get(int(phase_id), f"phase_{int(phase_id)}")

        if np.sum(mask) < 2:
            continue

        phase_timestamps = timestamps[mask]
        duration = (phase_timestamps[-1] - phase_timestamps[0]) * 1e-6 if phase_timestamps[0] > 1e10 else float(
            phase_timestamps[-1] - phase_timestamps[0])

        pos_errors = ds.get("mean_position_error", np.zeros(len(mask)))[mask]
        inf_times = ds.get("inference_time_us", np.zeros(len(mask)))[mask]
        saturation = ds.get("motor_saturation_ratio", np.zeros(len(mask)))[mask]
        out_var = ds.get("output_variance", np.zeros(len(mask)))[mask]
        out_roc = ds.get("output_rate_of_change", np.zeros(len(mask)))[mask]
        fallback = ds.get("fallback_active", np.zeros(len(mask), dtype=bool))[mask]

        # Per-axis position errors
        pos_ned = ds.get("position_error_ned", np.zeros((len(mask), 3)))
        if pos_ned.ndim == 2:
            pos_ned_phase = pos_ned[mask]
        else:
            pos_ned_phase = np.zeros((np.sum(mask), 3))

        # Angular velocity
        ang_vel = ds.get("angular_velocity_raw", np.zeros((len(mask), 3)))
        if ang_vel.ndim == 2:
            ang_vel_phase = ang_vel[mask]
        else:
            ang_vel_phase = np.zeros((np.sum(mask), 3))

        ang_vel_mag = np.linalg.norm(ang_vel_phase, axis=1)

        # Motor outputs
        motors = ds.get("motor_outputs", np.zeros((len(mask), 4)))
        if motors.ndim == 2:
            motors_phase = motors[mask]
        else:
            motors_phase = np.zeros((np.sum(mask), 4))

        stats = PhaseStatistics(
            phase_name=phase_name,
            duration_s=float(duration),
            sample_count=int(np.sum(mask)),
            mean_position_error=float(np.mean(pos_errors)) if len(pos_errors) > 0 else 0,
            max_position_error=float(np.max(pos_errors)) if len(pos_errors) > 0 else 0,
            std_position_error=float(np.std(pos_errors)) if len(pos_errors) > 0 else 0,
            mean_inference_time_us=float(np.mean(inf_times)) if len(inf_times) > 0 else 0,
            max_inference_time_us=float(np.max(inf_times)) if len(inf_times) > 0 else 0,
            mean_motor_saturation=float(np.mean(saturation)) if len(saturation) > 0 else 0,
            max_motor_saturation=float(np.max(saturation)) if len(saturation) > 0 else 0,
            mean_output_variance=float(np.mean(out_var)) if len(out_var) > 0 else 0,
            mean_output_rate_of_change=float(np.mean(out_roc)) if len(out_roc) > 0 else 0,
            mean_angular_velocity_magnitude=float(np.mean(ang_vel_mag)),
            max_angular_velocity_magnitude=float(np.max(ang_vel_mag)),
            fallback_count=int(np.sum(fallback)),
            mean_error_x=float(np.mean(pos_ned_phase[:, 0])) if pos_ned_phase.shape[0] > 0 else 0,
            mean_error_y=float(np.mean(pos_ned_phase[:, 1])) if pos_ned_phase.shape[0] > 0 else 0,
            mean_error_z=float(np.mean(pos_ned_phase[:, 2])) if pos_ned_phase.shape[0] > 0 else 0,
            motor_mean=[float(np.mean(motors_phase[:, i])) for i in range(4)] if motors_phase.shape[0] > 0 else [0] * 4,
            motor_std=[float(np.std(motors_phase[:, i])) for i in range(4)] if motors_phase.shape[0] > 0 else [0] * 4,
        )

        stats_list.append(stats)

    return stats_list


def compute_sim_real_gap_indicators(data: dict, sim_baseline: dict = None) -> dict:
    """
    Compute indicators of sim-to-real gap.
    These metrics highlight discrepancies between expected (simulated)
    and actual (real flight) performance.
    """
    ds = data.get("distillation_status", {})
    indicators = {}

    pos_errors = ds.get("mean_position_error", np.array([]))
    if len(pos_errors) > 0:
        # Steady-state error: error after settling (last 50% of data)
        half = len(pos_errors) // 2
        indicators["steady_state_error"] = float(np.mean(pos_errors[half:]))

        # Error trend: is performance degrading over time?
        if len(pos_errors) > 20:
            first_quarter = np.mean(pos_errors[:len(pos_errors) // 4])
            last_quarter = np.mean(pos_errors[3 * len(pos_errors) // 4:])
            indicators["error_drift"] = float(last_quarter - first_quarter)

    # Motor asymmetry: do motors behave differently? (indicates unmodeled dynamics)
    motors = ds.get("motor_outputs", np.zeros((0, 4)))
    if motors.ndim == 2 and motors.shape[0] > 10:
        motor_means = np.mean(motors, axis=0)
        indicators["motor_asymmetry"] = float(np.std(motor_means))
        indicators["motor_means"] = [float(m) for m in motor_means]

    # Angular velocity noise: higher than expected = unmodeled vibrations
    ang_vel = ds.get("angular_velocity_raw", np.zeros((0, 3)))
    if ang_vel.ndim == 2 and ang_vel.shape[0] > 10:
        ang_vel_std = np.std(ang_vel, axis=0)
        indicators["angular_velocity_noise"] = [float(s) for s in ang_vel_std]
        indicators["vibration_magnitude"] = float(np.mean(ang_vel_std))

    # Saturation frequency: how often are motors hitting limits?
    saturation = ds.get("motor_saturation_ratio", np.array([]))
    if len(saturation) > 0:
        indicators["saturation_frequency"] = float(np.mean(saturation > 0.5))
        indicators["mean_saturation"] = float(np.mean(saturation))

    # Compare with simulation baseline if provided
    if sim_baseline:
        for key in ["steady_state_error", "vibration_magnitude", "mean_saturation"]:
            if key in indicators and key in sim_baseline:
                indicators[f"{key}_gap"] = float(indicators[key] - sim_baseline[key])

    return indicators


def analyze_flight(
    log_path: str,
    sim_baseline: dict = None,
    output_format: str = "json",
) -> FlightAnalysisReport:
    """Run full flight analysis pipeline."""
    path = Path(log_path)

    if path.suffix == ".ulg":
        data = parse_ulog(log_path)
    else:
        data = parse_distillation_log(log_path)

    ds = data.get("distillation_status", {})

    # Build report
    report = FlightAnalysisReport(log_file=str(path.name))

    if "model_id" in ds and len(ds["model_id"]) > 0:
        report.model_id = int(ds["model_id"][0])

    timestamps = ds.get("timestamp", np.array([]))
    if len(timestamps) > 1:
        if timestamps[0] > 1e10:
            report.total_flight_time_s = float((timestamps[-1] - timestamps[0]) * 1e-6)
        else:
            report.total_flight_time_s = float(timestamps[-1] - timestamps[0])

    report.total_inferences = len(timestamps)

    success_rates = ds.get("success_rate", np.array([]))
    if len(success_rates) > 0:
        report.overall_success_rate = float(success_rates[-1])

    pos_errors = ds.get("mean_position_error", np.array([]))
    if len(pos_errors) > 0:
        report.overall_mean_position_error = float(np.mean(pos_errors))
        report.overall_max_position_error = float(np.max(pos_errors))

    fallback = ds.get("fallback_active", np.array([]))
    if len(fallback) > 0:
        report.fallback_activated = bool(np.any(fallback))

    # Phase statistics
    report.phase_statistics = [asdict(s) for s in compute_phase_statistics(data)]

    # Anomaly detection
    report.anomalies = [asdict(a) for a in detect_anomalies(data)]

    # Sim-to-real gap indicators
    report.sim_real_gap_indicators = compute_sim_real_gap_indicators(data, sim_baseline)

    return report


def format_report_for_llm(report: FlightAnalysisReport) -> str:
    """
    Format flight analysis report as a structured prompt for the LLM.
    This is the key interface between real-world data and LLM reasoning.
    """
    lines = [
        "# Real-World Flight Performance Report",
        "",
        f"Model ID: {report.model_id} (v{report.model_version})",
        f"Flight duration: {report.total_flight_time_s:.1f}s",
        f"Total inferences: {report.total_inferences}",
        f"Success rate: {report.overall_success_rate:.2%}",
        f"Overall position error: mean={report.overall_mean_position_error:.3f}m, max={report.overall_max_position_error:.3f}m",
        f"Fallback activated: {report.fallback_activated}",
        "",
    ]

    if report.phase_statistics:
        lines.append("## Per-Phase Performance")
        for phase in report.phase_statistics:
            lines.append(f"\n### {phase['phase_name']} ({phase['duration_s']:.1f}s, {phase['sample_count']} samples)")
            lines.append(f"- Position error: mean={phase['mean_position_error']:.3f}m, max={phase['max_position_error']:.3f}m, std={phase['std_position_error']:.3f}m")
            lines.append(f"- Per-axis error: X={phase['mean_error_x']:.3f}, Y={phase['mean_error_y']:.3f}, Z={phase['mean_error_z']:.3f}")
            lines.append(f"- Motor saturation: mean={phase['mean_motor_saturation']:.1%}, max={phase['max_motor_saturation']:.1%}")
            lines.append(f"- Angular velocity: mean={phase['mean_angular_velocity_magnitude']:.2f} rad/s, max={phase['max_angular_velocity_magnitude']:.2f} rad/s")
            lines.append(f"- Output smoothness: variance={phase['mean_output_variance']:.4f}, rate_of_change={phase['mean_output_rate_of_change']:.4f}")
            lines.append(f"- Inference time: mean={phase['mean_inference_time_us']:.0f}us, max={phase['max_inference_time_us']:.0f}us")
            if phase['fallback_count'] > 0:
                lines.append(f"- **Fallback events: {phase['fallback_count']}**")

    if report.anomalies:
        lines.append("\n## Detected Anomalies")
        for anomaly in report.anomalies:
            lines.append(f"- [{anomaly['severity'].upper()}] t={anomaly['timestamp_s']:.1f}s ({anomaly['flight_phase']}): {anomaly['description']}")

    if report.sim_real_gap_indicators:
        lines.append("\n## Sim-to-Real Gap Indicators")
        gap = report.sim_real_gap_indicators
        for key, value in gap.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            elif isinstance(value, list):
                lines.append(f"- {key}: [{', '.join(f'{v:.4f}' for v in value)}]")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze PX4 flight logs for distillation feedback")
    parser.add_argument("--log", type=str, required=True, help="Flight log file (.ulg, .json, .csv)")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    parser.add_argument("--sim-baseline", type=str, default=None, help="Simulation baseline metrics JSON")
    parser.add_argument("--format", type=str, choices=["json", "llm-prompt"], default="json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sim_baseline = None
    if args.sim_baseline:
        with open(args.sim_baseline) as f:
            sim_baseline = json.load(f)

    report = analyze_flight(args.log, sim_baseline)

    if args.format == "llm-prompt":
        output = format_report_for_llm(report)
        print(output)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
    else:
        report_dict = asdict(report)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report_dict, f, indent=2)
        else:
            print(json.dumps(report_dict, indent=2))


if __name__ == "__main__":
    main()
