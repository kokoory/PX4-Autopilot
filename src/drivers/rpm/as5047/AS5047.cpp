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

#include "AS5047.hpp"

/* Low-pass filter coefficient (alpha). Smaller = smoother, larger = faster response. */
static constexpr float LPF_ALPHA = 0.1f;

AS5047::AS5047(const I2CSPIDriverConfig &config) :
	SPI(config),
	ModuleParams(nullptr),
	I2CSPIDriver(config)
{
}

int AS5047::init()
{
	if (SPI::init() != PX4_OK) {
		PX4_ERR("SPI init failed");
		return PX4_ERROR;
	}

	PX4_DEBUG("poll: %" PRId32 " us, poles: %" PRId32,
		  _param_as5047_poll.get(),
		  _param_as5047_poles.get());

	ScheduleOnInterval(_param_as5047_poll.get());
	_rpm_pub.advertise();

	return PX4_OK;
}

int AS5047::probe()
{
	/* Read DIAAGC register to verify device is responding.
	 * A valid AS5047P will return a non-zero diagnostics/AGC value.
	 */
	uint16_t diag = 0;
	int ret = readRegister(AS5047_REG_DIAAGC, &diag);

	if (ret != PX4_OK) {
		PX4_DEBUG("probe: readRegister failed");
		return PX4_ERROR;
	}

	PX4_DEBUG("probe: DIAAGC = 0x%04x", diag);

	/* Check that AGC value (bits 0-7) is non-zero, which indicates
	 * a magnet is present and the device is functional.
	 */
	if ((diag & 0xFF) == 0) {
		PX4_DEBUG("probe: AGC value is zero, no magnet detected");
		return PX4_ERROR;
	}

	return PX4_OK;
}

uint16_t AS5047::addParity(uint16_t value)
{
	/* Even parity over bits 0-14 */
	uint16_t v = value & 0x7FFF;
	uint16_t parity = 0;

	for (int i = 0; i < 15; i++) {
		if (v & (1u << i)) {
			parity++;
		}
	}

	if (parity & 1u) {
		value |= AS5047_PARITY_BIT;

	} else {
		value &= ~AS5047_PARITY_BIT;
	}

	return value;
}

bool AS5047::checkParity(uint16_t frame)
{
	uint16_t parity = 0;

	for (int i = 0; i < 16; i++) {
		if (frame & (1u << i)) {
			parity++;
		}
	}

	return (parity & 1u) == 0; /* even parity: total set bits should be even */
}

int AS5047::readRegister(uint16_t reg, uint16_t *value)
{
	/* AS5047P SPI protocol:
	 * Frame 1 (send): command = READ_FLAG | address, with parity
	 * Frame 2 (send): NOP command (0x0000) to clock out the response
	 * The response to the register read comes back during frame 2.
	 */
	uint16_t cmd = addParity(AS5047_READ_FLAG | (reg & AS5047_DATA_MASK));
	uint16_t nop = addParity(AS5047_REG_NOP);
	uint16_t rx1 = 0;
	uint16_t rx2 = 0;

	/* Frame 1: send read command, response is from previous command (ignore) */
	int ret = transferhword(&cmd, &rx1, 1);

	if (ret != PX4_OK) {
		return PX4_ERROR;
	}

	/* Frame 2: send NOP, receive the actual register data */
	ret = transferhword(&nop, &rx2, 1);

	if (ret != PX4_OK) {
		return PX4_ERROR;
	}

	/* Check parity of received frame */
	if (!checkParity(rx2)) {
		PX4_DEBUG("readRegister: parity error, frame=0x%04x", rx2);
		return PX4_ERROR;
	}

	/* Check error flag (bit 14) */
	if (rx2 & AS5047_ERROR_FLAG) {
		PX4_DEBUG("readRegister: error flag set, frame=0x%04x", rx2);
		return PX4_ERROR;
	}

	*value = rx2 & AS5047_DATA_MASK;
	return PX4_OK;
}

int AS5047::readAngle(uint16_t *angle)
{
	return readRegister(AS5047_REG_ANGLECOM, angle);
}

void AS5047::RunImpl()
{
	uint16_t angle = 0;

	if (readAngle(&angle) != PX4_OK) {
		_error_count++;

		if (_error_count > 10) {
			PX4_ERR("AS5047 read errors: %d", _error_count);
			_error_count = 0;
		}

		return;
	}

	_error_count = 0;

	hrt_abstime now = hrt_absolute_time();

	if (_first_sample) {
		_prev_angle = angle;
		_prev_time = now;
		_first_sample = false;
		return;
	}

	/* Calculate delta time in microseconds */
	int32_t delta_time_us = static_cast<int32_t>(now - _prev_time);

	if (delta_time_us <= 0) {
		_prev_angle = angle;
		_prev_time = now;
		return;
	}

	/* Calculate delta angle with wraparound handling.
	 * The angle goes 0 -> 16383 -> 0 on each revolution.
	 * delta_angle can be positive or negative depending on direction.
	 * We use the shortest-path interpretation: if |delta| > 8192,
	 * the wrap went the other way.
	 */
	int32_t delta_angle = static_cast<int32_t>(angle) - static_cast<int32_t>(_prev_angle);

	/* Handle wraparound: choose the shorter path */
	if (delta_angle > (AS5047_RESOLUTION / 2)) {
		delta_angle -= AS5047_RESOLUTION;

	} else if (delta_angle < -(AS5047_RESOLUTION / 2)) {
		delta_angle += AS5047_RESOLUTION;
	}

	/* RPM calculation:
	 * mechanical_revolutions_per_second = (delta_angle / 16384) / (delta_time_us / 1e6)
	 * rpm_electrical = mechanical_rps * 60
	 * rpm_mechanical = rpm_electrical / pole_pairs
	 */
	float raw_rpm = (static_cast<float>(delta_angle) / static_cast<float>(AS5047_RESOLUTION))
			* (1e6f / static_cast<float>(delta_time_us))
			* 60.0f;

	/* Divide by number of magnetic pole pairs to get mechanical RPM */
	int32_t poles = _param_as5047_poles.get();

	if (poles < 1) {
		poles = 1;
	}

	raw_rpm /= static_cast<float>(poles);

	/* Apply low-pass filter */
	_filtered_rpm = _filtered_rpm + LPF_ALPHA * (raw_rpm - _filtered_rpm);

	/* Zero RPM detection: if no angle change, report zero */
	if (delta_angle == 0) {
		/* Decay the filtered value towards zero */
		_filtered_rpm *= (1.0f - LPF_ALPHA);
	}

	/* Publish to uORB */
	rpm_s msg{};
	msg.rpm_estimate = _filtered_rpm;
	msg.rpm_raw = raw_rpm;
	msg.timestamp = hrt_absolute_time();
	_rpm_pub.publish(msg);

	_prev_angle = angle;
	_prev_time = now;
}

void AS5047::print_status()
{
	I2CSPIDriverBase::print_status();
	PX4_INFO("poll interval:  %" PRId32 " us", _param_as5047_poll.get());
	PX4_INFO("pole pairs:     %" PRId32, _param_as5047_poles.get());
	PX4_INFO("filtered RPM:   %.1f", (double)_filtered_rpm);
	PX4_INFO("error count:    %d", _error_count);
}
