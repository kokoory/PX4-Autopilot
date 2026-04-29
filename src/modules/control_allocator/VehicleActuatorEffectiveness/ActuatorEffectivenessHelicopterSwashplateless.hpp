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

#pragma once

#include "control_allocation/actuator_effectiveness/ActuatorEffectiveness.hpp"

#include <px4_platform_common/module_params.h>

#include <uORB/Subscription.hpp>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/manual_control_switches.h>
#include <uORB/topics/rotor_position.h>

class ActuatorEffectivenessHelicopterSwashplateless : public ModuleParams, public ActuatorEffectiveness
{
public:

	static constexpr int NUM_CURVE_POINTS = 5;

	struct Geometry {
		float throttle_curve[NUM_CURVE_POINTS];
		float yaw_sign;
		float yaw_throttle_scale;
		float spoolup_time;
		float phase_offset; // degrees, to compensate for blade lag
	};

	ActuatorEffectivenessHelicopterSwashplateless(ModuleParams *parent);
	virtual ~ActuatorEffectivenessHelicopterSwashplateless() = default;

	bool getEffectivenessMatrix(Configuration &configuration, EffectivenessUpdateReason external_update) override;

	const char *name() const override { return "Helicopter Swashplateless"; }

	const Geometry &geometry() const { return _geometry; }

	void updateSetpoint(const matrix::Vector<float, NUM_AXES> &control_sp, int matrix_index, ActuatorVector &actuator_sp,
			    const ActuatorVector &actuator_min, const ActuatorVector &actuator_max) override;

	void getUnallocatedControl(int matrix_index, control_allocator_status_s &status) override;
private:
	float throttleSpoolupProgress();
	bool mainMotorEnaged();

	void updateParams() override;

	struct SaturationFlags {
		bool roll_pos;
		bool roll_neg;
		bool pitch_pos;
		bool pitch_neg;
		bool yaw_pos;
		bool yaw_neg;
		bool thrust_pos;
		bool thrust_neg;
	};
	static void setSaturationFlag(float coeff, bool &positive_flag, bool &negative_flag);

	struct ParamHandles {
		param_t throttle_curve[NUM_CURVE_POINTS];
		param_t yaw_throttle_scale;
		param_t yaw_ccw;
		param_t spoolup_time;
		param_t phase_offset;
	};
	ParamHandles _param_handles{};

	Geometry _geometry{};

	SaturationFlags _saturation_flags;

	// Rotor position from encoder (e.g. AS5047)
	uORB::Subscription _rotor_pos_sub{ORB_ID(rotor_position)};
	float _rotor_angle{0.f};
	bool _rotor_valid{false};

	// Throttle spoolup state
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	bool _armed{false};
	uint64_t _armed_time{0};

	uORB::Subscription _manual_control_switches_sub{ORB_ID(manual_control_switches)};
	bool _main_motor_engaged{true};
};
