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

#include "AS5048B.hpp"

#include <math.h>

AS5048B::AS5048B(const I2CSPIDriverConfig &config) :
	I2C(config),
	ModuleParams(nullptr),
	I2CSPIDriver(config)
{
}

int AS5048B::init()
{
	if (I2C::init() != PX4_OK) {
		return PX4_ERROR;
	}

	PX4_DEBUG("addr: 0x%02x, poll: %" PRId32 " us, poles: %" PRId32,
		  get_device_address(),
		  _param_as5048b_poll.get(),
		  _param_as5048b_poles.get());

	_last_measurement_time = hrt_absolute_time();
	_last_motion_time = _last_measurement_time;
	_last_angle = -1;

	ScheduleOnInterval(_param_as5048b_poll.get());
	_rpm_pub.advertise();

	return PX4_OK;
}

int AS5048B::probe()
{
	// Read the AGC register to verify the device is responding
	uint8_t reg = AS5048B_REG_AGC;
	uint8_t val{};
	int ret = transfer(&reg, 1, &val, 1);

	if (ret != PX4_OK) {
		PX4_DEBUG("probe: AGC register read failed, ret=%d", ret);
		return PX4_ERROR;
	}

	// AGC value should be non-zero when a magnet is present,
	// but we accept any successful transfer as proof the device exists
	PX4_DEBUG("probe: AGC=0x%02x", val);

	// Also verify we can read the angle register
	reg = AS5048B_REG_ANGLE_HIGH;
	ret = transfer(&reg, 1, &val, 1);

	if (ret != PX4_OK) {
		PX4_DEBUG("probe: angle register read failed, ret=%d", ret);
		return PX4_ERROR;
	}

	return PX4_OK;
}

uint8_t AS5048B::readRegister(uint8_t reg)
{
	uint8_t rcv{};
	int ret = transfer(&reg, 1, &rcv, 1);

	if (PX4_OK != ret) {
		PX4_DEBUG("readRegister(0x%02x): i2c::transfer returned %d", reg, ret);
		_transfer_fail_count++;
	}

	return rcv;
}

uint16_t AS5048B::readAngle()
{
	uint8_t high = readRegister(AS5048B_REG_ANGLE_HIGH);
	uint8_t low  = readRegister(AS5048B_REG_ANGLE_LOW);

	// 14-bit angle: high byte contains bits 13:6, low byte bits 5:0 (upper 2 bits unused)
	uint16_t angle = ((uint16_t)high << 6) | (low & 0x3F);

	return angle;
}

void AS5048B::RunImpl()
{
	hrt_abstime now = hrt_absolute_time();
	int32_t delta_time_us = now - _last_measurement_time;

	// Guard against being called too early
	if (delta_time_us < _param_as5048b_poll.get() / 2) {
		return;
	}

	// Read the current angle
	uint16_t current_angle = readAngle();

	// Check for I2C transfer errors
	if (_transfer_fail_count > 0) {
		PX4_ERR("as5048b I2C transfer failures: %d, reinitializing", _transfer_fail_count);
		_transfer_fail_count = 0;
		_last_angle = -1;
		_last_measurement_time = now;
		return;
	}

	// First valid reading: store and return
	if (_last_angle < 0) {
		_last_angle = (int16_t)current_angle;
		_last_measurement_time = now;
		return;
	}

	// Calculate angle delta with wraparound handling
	int32_t delta_angle = (int32_t)current_angle - (int32_t)_last_angle;

	// Handle wraparound: take the shortest path
	if (delta_angle > (AS5048B_RESOLUTION / 2)) {
		delta_angle -= AS5048B_RESOLUTION;

	} else if (delta_angle < -(AS5048B_RESOLUTION / 2)) {
		delta_angle += AS5048B_RESOLUTION;
	}

	// Calculate raw RPM from angle delta
	// rpm = (delta_angle / full_revolution) * (1e6 / delta_time_us) * 60
	// Divide by pole pairs to get mechanical RPM
	float raw_rpm = 0.0f;

	if (delta_time_us > 0) {
		raw_rpm = ((float)delta_angle / (float)AS5048B_RESOLUTION)
			  * (1e6f / (float)delta_time_us)
			  * 60.0f
			  / (float)_param_as5048b_poles.get();
	}

	// Apply exponential moving average filter
	_filtered_rpm = AS5048B_ALPHA_DEFAULT * raw_rpm + (1.0f - AS5048B_ALPHA_DEFAULT) * _filtered_rpm;

	// Track last time significant motion was detected
	if (fabsf(raw_rpm) > 0.5f) {
		_last_motion_time = now;
	}

	// Zero RPM detection: if no movement for timeout period, force zero
	if ((now - _last_motion_time) > AS5048B_ZERO_RPM_TIMEOUT_US) {
		_filtered_rpm = 0.0f;
	}

	// Update state
	_last_angle = (int16_t)current_angle;
	_last_measurement_time = now;

	// Publish to uORB
	rpm_s msg{};
	msg.rpm_estimate = _filtered_rpm;
	msg.rpm_raw = raw_rpm;
	msg.timestamp = hrt_absolute_time();
	_rpm_pub.publish(msg);
}

void AS5048B::print_status()
{
	I2CSPIDriverBase::print_status();
	PX4_INFO("poll interval:  %" PRId32 " us", _param_as5048b_poll.get());
	PX4_INFO("pole pairs:     %" PRId32, _param_as5048b_poles.get());
	PX4_INFO("filtered RPM:   %.1f", (double)_filtered_rpm);
	PX4_INFO("last angle:     %d / %d", (int)_last_angle, AS5048B_RESOLUTION);
}
