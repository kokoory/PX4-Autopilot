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

/**
 * @file AS5047.hpp
 *
 * Driver for the AS5047P magnetic rotary position sensor (SPI).
 * Reads 14-bit absolute angle and computes motor RPM.
 */

#pragma once

#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/i2c_spi_buses.h>
#include <drivers/device/spi.h>
#include <uORB/Publication.hpp>
#include <uORB/PublicationMulti.hpp>
#include <uORB/topics/rpm.h>
#include <uORB/topics/rotor_position.h>
#include <drivers/drv_hrt.h>

/* AS5047P register addresses */
#define AS5047_REG_NOP         0x0000
#define AS5047_REG_ERRFL       0x0001
#define AS5047_REG_DIAAGC      0x3FFC
#define AS5047_REG_ANGLECOM    0x3FFF

/* AS5047P SPI frame bits */
#define AS5047_READ_FLAG       (1u << 14)   /* Bit 14: read = 1, write = 0 */
#define AS5047_PARITY_BIT      (1u << 15)   /* Bit 15: even parity */
#define AS5047_DATA_MASK       0x3FFFu      /* Bits 0-13: data / angle */
#define AS5047_ERROR_FLAG      (1u << 14)   /* Bit 14 of response: error flag */
#define AS5047_RESOLUTION      16384        /* 14-bit: 0-16383 */

class AS5047 : public device::SPI, public ModuleParams, public I2CSPIDriver<AS5047>
{
public:
	AS5047(const I2CSPIDriverConfig &config);
	~AS5047() override = default;

	static void print_usage();

	void RunImpl();

	int    init() override;
	void   print_status() override;

private:
	int probe() override;

	/**
	 * Calculate even parity for bits 0-14 of a 16-bit SPI frame.
	 * @param value  The 15-bit value (bits 0-14).
	 * @return       The value with parity bit set in bit 15.
	 */
	static uint16_t addParity(uint16_t value);

	/**
	 * Check even parity of a received 16-bit SPI frame.
	 * @param frame  The full 16-bit received frame.
	 * @return       true if parity is correct (even).
	 */
	static bool checkParity(uint16_t frame);

	/**
	 * Perform a 16-bit SPI read of the given register address.
	 * AS5047P SPI protocol: send read command, then send NOP to clock out
	 * the response on the next frame.
	 * @param reg    Register address (14-bit).
	 * @param value  Pointer to store the 14-bit result.
	 * @return       PX4_OK on success, PX4_ERROR on failure.
	 */
	int readRegister(uint16_t reg, uint16_t *value);

	/**
	 * Read the 14-bit angle from ANGLECOM register.
	 * @param angle  Pointer to store the angle (0-16383).
	 * @return       PX4_OK on success, PX4_ERROR on failure.
	 */
	int readAngle(uint16_t *angle);

	/* State for RPM calculation */
	uint16_t    _prev_angle{0};
	hrt_abstime _prev_time{0};
	bool        _first_sample{true};
	float       _filtered_rpm{0.0f};
	int         _error_count{0};

	uORB::PublicationMulti<rpm_s> _rpm_pub{ORB_ID(rpm)};
	uORB::Publication<rotor_position_s> _rotor_pos_pub{ORB_ID(rotor_position)};

	float       _current_angle_rad{0.0f};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::AS5047_POLL>)  _param_as5047_poll,
		(ParamInt<px4::params::AS5047_POLES>) _param_as5047_poles
	)
};
