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
 * @file AS5048B.hpp
 *
 * Driver for RPM measurement using AS5048B 14-bit magnetic rotary encoder
 * over I2C interface.
 */

#pragma once

#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/i2c_spi_buses.h>
#include <drivers/device/i2c.h>
#include <uORB/Publication.hpp>
#include <uORB/PublicationMulti.hpp>
#include <uORB/topics/rpm.h>
#include <drivers/drv_hrt.h>

/* Default I2C address (A1=0, A2=0). Range: 0x40-0x43 */
#define AS5048B_BASEADDR_DEFAULT        0x40

/* AS5048B Register Map */
#define AS5048B_REG_AGC                 0xFA    /* Automatic Gain Control (diagnostic) */
#define AS5048B_REG_MAGNITUDE_HIGH      0xFC    /* Magnitude high byte */
#define AS5048B_REG_MAGNITUDE_LOW       0xFD    /* Magnitude low byte */
#define AS5048B_REG_ANGLE_HIGH          0xFE    /* Angle bits 13:6 (8 bits) */
#define AS5048B_REG_ANGLE_LOW           0xFF    /* Angle bits 5:0 (6 bits, upper 2 unused) */

/* 14-bit angle resolution: 0-16383 maps to 0-360 degrees */
#define AS5048B_RESOLUTION              16384
#define AS5048B_ANGLE_LOW_MASK          0x3F    /* Mask for lower 6 bits */

/* RPM calculation constants */
#define AS5048B_ALPHA_DEFAULT           0.1f    /* Low-pass filter coefficient */
#define AS5048B_ZERO_RPM_TIMEOUT_US     500000  /* 500ms timeout for zero RPM detection */

class AS5048B : public device::I2C, public ModuleParams, public I2CSPIDriver<AS5048B>
{
public:
	AS5048B(const I2CSPIDriverConfig &config);
	~AS5048B() override = default;

	static void print_usage();

	void		RunImpl();

	int    init() override;
	void   print_status() override;

private:

	int  probe() override;

	uint8_t        readRegister(uint8_t reg);
	uint16_t       readAngle();

	int16_t        _last_angle{-1};
	hrt_abstime    _last_measurement_time{0};
	hrt_abstime    _last_motion_time{0};
	int            _transfer_fail_count{0};

	float          _filtered_rpm{0.0f};

	uORB::PublicationMulti<rpm_s> _rpm_pub{ORB_ID(rpm)};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::AS5048B_POLL>) _param_as5048b_poll,
		(ParamInt<px4::params::AS5048B_POLES>) _param_as5048b_poles
	)
};
