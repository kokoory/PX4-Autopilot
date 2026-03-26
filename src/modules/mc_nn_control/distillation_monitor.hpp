/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file distillation_monitor.hpp
 * Safety monitor for knowledge-distilled neural network flight controllers.
 *
 * Validates NN inference outputs and manages automatic fallback to
 * traditional PID controllers when the distilled model produces
 * invalid or unsafe outputs.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <uORB/topics/distillation_status.h>

class DistillationMonitor
{
public:
	DistillationMonitor() = default;
	~DistillationMonitor() = default;

	/**
	 * Validate neural network output for safety.
	 * Checks for NaN values and out-of-range outputs.
	 *
	 * @param output Motor command array (4 elements)
	 * @param num_outputs Number of output elements
	 * @return true if output is valid, false otherwise
	 */
	bool validate_output(const float *output, int num_outputs = 4)
	{
		for (int i = 0; i < num_outputs; i++) {
			if (!std::isfinite(output[i])) {
				_last_fallback_reason = distillation_status_s::FALLBACK_OUTPUT_NAN;
				record_failure();
				return false;
			}

			if (output[i] < -1.5f || output[i] > 1.5f) {
				_last_fallback_reason = distillation_status_s::FALLBACK_OUTPUT_RANGE;
				record_failure();
				return false;
			}
		}

		record_success();
		update_output_variance(output, num_outputs);
		return true;
	}

	/**
	 * Check if inference time is within acceptable bounds.
	 *
	 * @param inference_time_us Inference time in microseconds
	 * @param max_time_us Maximum allowed inference time
	 * @return true if timing is acceptable
	 */
	bool check_timing(int32_t inference_time_us, int32_t max_time_us)
	{
		_last_inference_time_us = inference_time_us;

		if (inference_time_us > max_time_us) {
			_last_fallback_reason = distillation_status_s::FALLBACK_INFERENCE_TIMEOUT;
			record_failure();
			return false;
		}

		return true;
	}

	/**
	 * Determine if the controller should fall back to PID.
	 *
	 * @param error_limit Maximum consecutive errors before fallback
	 * @return true if fallback should be activated
	 */
	bool should_fallback(int32_t error_limit) const
	{
		return _consecutive_errors >= error_limit;
	}

	/**
	 * Reset the monitor state (e.g., when switching modes).
	 */
	void reset()
	{
		_consecutive_errors = 0;
		_fallback_active = false;
		_last_fallback_reason = distillation_status_s::FALLBACK_NONE;
	}

	/**
	 * Set fallback state manually (e.g., from parameter change).
	 */
	void set_fallback(bool active, uint8_t reason = distillation_status_s::FALLBACK_MANUAL)
	{
		_fallback_active = active;

		if (active) {
			_last_fallback_reason = reason;
		} else {
			_last_fallback_reason = distillation_status_s::FALLBACK_NONE;
			_consecutive_errors = 0;
		}
	}

	/**
	 * Update position error tracking.
	 */
	void update_position_error(float error)
	{
		constexpr float alpha = 0.05f;  // Exponential moving average
		_mean_position_error = alpha * error + (1.0f - alpha) * _mean_position_error;
	}

	/**
	 * Update runtime feedback diagnostics for sim-to-real gap analysis.
	 * Called each control cycle with current motor outputs and state data.
	 */
	void update_feedback_diagnostics(const float *motor_outputs, int num_motors,
					 const float *pos_error_ned, const float *angular_vel)
	{
		// Store latest motor outputs
		for (int i = 0; i < 4 && i < num_motors; i++) {
			// Compute rate of change
			float delta = motor_outputs[i] - _prev_motor_outputs[i];
			_motor_rate_of_change += delta * delta;
			_prev_motor_outputs[i] = motor_outputs[i];
		}

		_motor_rate_of_change = sqrtf(_motor_rate_of_change / 4.0f);

		// Motor saturation: count how many motors are near limits
		int saturated = 0;

		for (int i = 0; i < 4 && i < num_motors; i++) {
			if (motor_outputs[i] > 0.95f || motor_outputs[i] < -0.95f) {
				saturated++;
			}
		}

		constexpr float sat_alpha = 0.1f;
		_motor_saturation_ratio = sat_alpha * (static_cast<float>(saturated) / 4.0f)
					  + (1.0f - sat_alpha) * _motor_saturation_ratio;

		// Store position error and angular velocity for the status message
		for (int i = 0; i < 3; i++) {
			_last_pos_error_ned[i] = pos_error_ned[i];
			_last_angular_vel[i] = angular_vel[i];
		}

		// Detect flight phase from velocity and position error magnitude
		float vel_magnitude = sqrtf(angular_vel[0] * angular_vel[0] +
					    angular_vel[1] * angular_vel[1] +
					    angular_vel[2] * angular_vel[2]);
		float pos_err_mag = sqrtf(pos_error_ned[0] * pos_error_ned[0] +
					  pos_error_ned[1] * pos_error_ned[1] +
					  pos_error_ned[2] * pos_error_ned[2]);

		if (pos_err_mag < 0.3f && vel_magnitude < 0.5f) {
			_flight_phase = distillation_status_s::PHASE_HOVER;

		} else if (vel_magnitude > 2.0f) {
			_flight_phase = distillation_status_s::PHASE_MANEUVER;

		} else {
			_flight_phase = distillation_status_s::PHASE_CRUISE;
		}
	}

	/**
	 * Populate a DistillationStatus message with current monitor state.
	 */
	void populate_status(distillation_status_s &status, uint8_t model_id, uint8_t model_version) const
	{
		status.model_id = model_id;
		status.model_version = model_version;
		status.inference_time_us = _last_inference_time_us;
		status.controller_time_us = 0;  // Set by caller
		status.output_variance = _output_variance;
		status.mean_position_error = _mean_position_error;
		status.fallback_active = _fallback_active;
		status.fallback_reason = _last_fallback_reason;
		status.total_inferences = _total_inferences;
		status.failed_inferences = _failed_inferences;
		status.success_rate = (_total_inferences > 0)
				      ? static_cast<float>(_total_inferences - _failed_inferences) / static_cast<float>(_total_inferences)
				      : 1.0f;

		// Runtime feedback diagnostics
		for (int i = 0; i < 4; i++) {
			status.motor_outputs[i] = _prev_motor_outputs[i];
		}

		for (int i = 0; i < 3; i++) {
			status.position_error_ned[i] = _last_pos_error_ned[i];
			status.angular_velocity_raw[i] = _last_angular_vel[i];
		}

		status.motor_saturation_ratio = _motor_saturation_ratio;
		status.output_rate_of_change = _motor_rate_of_change;
		status.flight_phase = _flight_phase;
	}

	// Accessors
	bool fallback_active() const { return _fallback_active; }
	uint32_t total_inferences() const { return _total_inferences; }
	uint32_t failed_inferences() const { return _failed_inferences; }
	uint8_t last_fallback_reason() const { return _last_fallback_reason; }
	float output_variance() const { return _output_variance; }
	float motor_saturation_ratio() const { return _motor_saturation_ratio; }
	uint8_t flight_phase() const { return _flight_phase; }

private:
	void record_success()
	{
		_total_inferences++;
		_consecutive_errors = 0;
	}

	void record_failure()
	{
		_total_inferences++;
		_failed_inferences++;
		_consecutive_errors++;
	}

	void update_output_variance(const float *output, int num_outputs)
	{
		float mean = 0.0f;

		for (int i = 0; i < num_outputs; i++) {
			mean += output[i];
		}

		mean /= static_cast<float>(num_outputs);

		float variance = 0.0f;

		for (int i = 0; i < num_outputs; i++) {
			float diff = output[i] - mean;
			variance += diff * diff;
		}

		variance /= static_cast<float>(num_outputs);

		// Exponential moving average of variance
		constexpr float alpha = 0.1f;
		_output_variance = alpha * variance + (1.0f - alpha) * _output_variance;
	}

	uint32_t _total_inferences{0};
	uint32_t _failed_inferences{0};
	int32_t _consecutive_errors{0};
	int32_t _last_inference_time_us{0};

	bool _fallback_active{false};
	uint8_t _last_fallback_reason{distillation_status_s::FALLBACK_NONE};

	float _output_variance{0.0f};
	float _mean_position_error{0.0f};

	// Runtime feedback diagnostics
	float _prev_motor_outputs[4]{0.0f, 0.0f, 0.0f, 0.0f};
	float _last_pos_error_ned[3]{0.0f, 0.0f, 0.0f};
	float _last_angular_vel[3]{0.0f, 0.0f, 0.0f};
	float _motor_saturation_ratio{0.0f};
	float _motor_rate_of_change{0.0f};
	uint8_t _flight_phase{distillation_status_s::PHASE_IDLE};
};
