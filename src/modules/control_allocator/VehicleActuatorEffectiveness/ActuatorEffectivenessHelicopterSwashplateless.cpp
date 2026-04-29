/****************************************************************************
 *
 *   Copyright (c) 2024 PX4 Development Team. All rights reserved.
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

#include "ActuatorEffectivenessHelicopterSwashplateless.hpp"
#include <lib/mathlib/mathlib.h>

using namespace matrix;
using namespace time_literals;

ActuatorEffectivenessHelicopterSwashplateless::ActuatorEffectivenessHelicopterSwashplateless(ModuleParams *parent)
	: ModuleParams(parent)
{
	for (int i = 0; i < NUM_CURVE_POINTS; ++i) {
		char buffer[17];
		snprintf(buffer, sizeof(buffer), "CA_HELI_THR_C%u", i);
		_param_handles.throttle_curve[i] = param_find(buffer);
	}

	_param_handles.yaw_throttle_scale = param_find("CA_HELI_YAW_TH_S");
	_param_handles.yaw_ccw = param_find("CA_HELI_YAW_CCW");
	_param_handles.spoolup_time = param_find("COM_SPOOLUP_TIME");
	_param_handles.phase_offset = param_find("CA_SWL_PHASE");

	updateParams();
}

void ActuatorEffectivenessHelicopterSwashplateless::updateParams()
{
	ModuleParams::updateParams();

	for (int i = 0; i < NUM_CURVE_POINTS; ++i) {
		param_get(_param_handles.throttle_curve[i], &_geometry.throttle_curve[i]);
	}

	param_get(_param_handles.yaw_throttle_scale, &_geometry.yaw_throttle_scale);
	param_get(_param_handles.spoolup_time, &_geometry.spoolup_time);

	int32_t yaw_ccw = 0;
	param_get(_param_handles.yaw_ccw, &yaw_ccw);
	_geometry.yaw_sign = (yaw_ccw == 1) ? -1.f : 1.f;

	float phase_offset_deg = 0.f;
	param_get(_param_handles.phase_offset, &phase_offset_deg);
	_geometry.phase_offset = phase_offset_deg;
}

bool ActuatorEffectivenessHelicopterSwashplateless::getEffectivenessMatrix(Configuration &configuration,
		EffectivenessUpdateReason external_update)
{
	if (external_update == EffectivenessUpdateReason::NO_EXTERNAL_UPDATE) {
		return false;
	}

	// Main motor (thrust) - non-linear, handled in updateSetpoint()
	configuration.addActuator(ActuatorType::MOTORS, Vector3f{}, Vector3f{});

	// Tail motor (yaw)
	configuration.addActuator(ActuatorType::MOTORS, Vector3f{}, Vector3f{});

	return true;
}

void ActuatorEffectivenessHelicopterSwashplateless::updateSetpoint(const matrix::Vector<float, NUM_AXES> &control_sp,
		int matrix_index, ActuatorVector &actuator_sp, const ActuatorVector &actuator_min,
		const ActuatorVector &actuator_max)
{
	_saturation_flags = {};

	const float spoolup_progress = throttleSpoolupProgress();

	// Read rotor position from encoder
	rotor_position_s rotor_pos;

	if (_rotor_pos_sub.update(&rotor_pos)) {
		_rotor_angle = rotor_pos.angle_rad;
		_rotor_valid = rotor_pos.valid;
	}

	// Throttle curve
	const float base_throttle = math::interpolateN(-control_sp(ControlAxis::THRUST_Z),
				    _geometry.throttle_curve) * spoolup_progress;

	// Sinusoidal motor speed modulation for cyclic control
	float cyclic_modulation = 0.f;

	if (_rotor_valid && spoolup_progress > 0.5f) {
		const float phase_rad = math::radians(_geometry.phase_offset);
		const float roll_cmd = control_sp(ControlAxis::ROLL);
		const float pitch_cmd = control_sp(ControlAxis::PITCH);

		// Modulate motor speed sinusoidally synced to rotor angle
		// Roll: sin(angle + phase), Pitch: cos(angle + phase)
		cyclic_modulation = -roll_cmd * sinf(_rotor_angle + phase_rad)
				    + pitch_cmd * cosf(_rotor_angle + phase_rad);

		// Scale modulation to a reasonable range
		cyclic_modulation = math::constrain(cyclic_modulation, -0.3f, 0.3f);
	}

	// Main motor: base throttle + cyclic modulation
	actuator_sp(0) = mainMotorEnaged() ? math::constrain(base_throttle + cyclic_modulation, 0.f, 1.f) : NAN;

	// Tail motor: yaw control + throttle feedforward
	actuator_sp(1) = control_sp(ControlAxis::YAW) * _geometry.yaw_sign
			 + base_throttle * _geometry.yaw_throttle_scale;

	// Saturation checks
	if (actuator_sp(0) <= actuator_min(0)) {
		_saturation_flags.thrust_neg = true;

	} else if (actuator_sp(0) >= actuator_max(0)) {
		_saturation_flags.thrust_pos = true;
	}

	if (actuator_sp(1) < actuator_min(1)) {
		setSaturationFlag(_geometry.yaw_sign, _saturation_flags.yaw_neg, _saturation_flags.yaw_pos);

	} else if (actuator_sp(1) > actuator_max(1)) {
		setSaturationFlag(_geometry.yaw_sign, _saturation_flags.yaw_pos, _saturation_flags.yaw_neg);
	}
}

bool ActuatorEffectivenessHelicopterSwashplateless::mainMotorEnaged()
{
	manual_control_switches_s manual_control_switches;

	if (_manual_control_switches_sub.update(&manual_control_switches)) {
		_main_motor_engaged = manual_control_switches.engage_main_motor_switch == manual_control_switches_s::SWITCH_POS_NONE
				      || manual_control_switches.engage_main_motor_switch == manual_control_switches_s::SWITCH_POS_ON;
	}

	return _main_motor_engaged;
}

float ActuatorEffectivenessHelicopterSwashplateless::throttleSpoolupProgress()
{
	vehicle_status_s vehicle_status;

	if (_vehicle_status_sub.update(&vehicle_status)) {
		_armed = vehicle_status.arming_state == vehicle_status_s::ARMING_STATE_ARMED;
		_armed_time = vehicle_status.armed_time;
	}

	const float time_since_arming = (hrt_absolute_time() - _armed_time) / 1e6f;
	const float spoolup_progress = time_since_arming / _geometry.spoolup_time;

	if (_armed && spoolup_progress < 1.f) {
		return spoolup_progress;
	}

	return 1.f;
}

void ActuatorEffectivenessHelicopterSwashplateless::setSaturationFlag(float coeff, bool &positive_flag,
		bool &negative_flag)
{
	if (coeff > 0.f) {
		positive_flag = true;

	} else if (coeff < 0.f) {
		negative_flag = true;
	}
}

void ActuatorEffectivenessHelicopterSwashplateless::getUnallocatedControl(int matrix_index,
		control_allocator_status_s &status)
{
	if (_saturation_flags.roll_pos) {
		status.unallocated_torque[0] = 1.f;

	} else if (_saturation_flags.roll_neg) {
		status.unallocated_torque[0] = -1.f;

	} else {
		status.unallocated_torque[0] = 0.f;
	}

	if (_saturation_flags.pitch_pos) {
		status.unallocated_torque[1] = 1.f;

	} else if (_saturation_flags.pitch_neg) {
		status.unallocated_torque[1] = -1.f;

	} else {
		status.unallocated_torque[1] = 0.f;
	}

	if (_saturation_flags.yaw_pos) {
		status.unallocated_torque[2] = 1.f;

	} else if (_saturation_flags.yaw_neg) {
		status.unallocated_torque[2] = -1.f;

	} else {
		status.unallocated_torque[2] = 0.f;
	}

	if (_saturation_flags.thrust_pos) {
		status.unallocated_thrust[2] = 1.f;

	} else if (_saturation_flags.thrust_neg) {
		status.unallocated_thrust[2] = -1.f;

	} else {
		status.unallocated_thrust[2] = 0.f;
	}
}
