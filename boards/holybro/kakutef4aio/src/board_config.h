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
 * @file board_config.h
 *
 * Holybro Kakute F4 AIO V2.1 internal definitions
 *
 * Singlecopter configuration:
 *   M1-M4: Servo vanes (PWM)
 *   LED pad (PC8): Motor ESC (repurposed)
 */

#pragma once

#include <nuttx/compiler.h>
#include <stdint.h>

/* LEDs - PB5 (active low) */
#define GPIO_LED1        /* PB5 */  (GPIO_OUTPUT|GPIO_OPENDRAIN|GPIO_SPEED_50MHz|GPIO_OUTPUT_SET|GPIO_PORTB|GPIO_PIN5)
#define GPIO_LED_BLUE    GPIO_LED1

#define BOARD_OVERLOAD_LED     LED_BLUE

/* ADC channels
 * STM32F4 naming: GPIO_ADC1_INxx
 */
#define ADC1_CH(n)                  (n)

#define ADC_BATTERY_VOLTAGE_CHANNEL        /* PC3 */  ADC1_CH(13)
#define ADC_BATTERY_CURRENT_CHANNEL        /* PC2 */  ADC1_CH(12)
#define ADC_RSSI_IN_CHANNEL                /* PC1 */  ADC1_CH(11)

#define ADC_CHANNELS \
	((1 << ADC_BATTERY_VOLTAGE_CHANNEL)       | \
	 (1 << ADC_BATTERY_CURRENT_CHANNEL)       | \
	 (1 << ADC_RSSI_IN_CHANNEL))

#define BOARD_ADC_OPEN_CIRCUIT_V     (5.6f)

/* PWM - 5 outputs: M1-M4 (servos) + LED pad (motor) */
#define DIRECT_PWM_OUTPUT_CHANNELS  5
#define BOARD_NUM_IO_TIMERS 3
#define BOARD_HAS_PWM    DIRECT_PWM_OUTPUT_CHANNELS

/* Tone alarm output - PC9 (buzzer) */
#define GPIO_TONE_ALARM_IDLE    /* PC9 */ (GPIO_OUTPUT|GPIO_PUSHPULL|GPIO_SPEED_2MHz|GPIO_OUTPUT_CLEAR|GPIO_PORTC|GPIO_PIN9)
#define GPIO_TONE_ALARM_GPIO    /* PC9 */ (GPIO_OUTPUT|GPIO_PUSHPULL|GPIO_SPEED_2MHz|GPIO_OUTPUT_SET|GPIO_PORTC|GPIO_PIN9)

/* USB OTG FS */
#define GPIO_OTGFS_VBUS         /* PA8 */ (GPIO_INPUT|GPIO_PULLDOWN|GPIO_SPEED_100MHz|GPIO_PORTA|GPIO_PIN8)

/* High-resolution timer */
#define HRT_TIMER               4  /* use timer4 for the HRT */
#define HRT_TIMER_CHANNEL       1  /* use capture/compare channel 1 */

/* RC Serial port */
#define RC_SERIAL_PORT          "/dev/ttyS2"

/* SBUS inversion */
#define GPIO_SBUS_INV           /* PB15 */ (GPIO_OUTPUT|GPIO_PUSHPULL|GPIO_OUTPUT_CLEAR|GPIO_PORTB|GPIO_PIN15)
#define INVERT_RC_INPUT(_invert_true)  px4_arch_gpiowrite(GPIO_SBUS_INV, _invert_true)

/* Power switch controls */
#define BOARD_ADC_USB_CONNECTED (px4_arch_gpioread(GPIO_OTGFS_VBUS))
#define BOARD_ADC_SERVO_VALID     (1)
#define BOARD_ADC_BRICK1_VALID  (1)

/* This board provides a DMA pool and APIs */
#define BOARD_DMA_ALLOC_POOL_SIZE 5120

#define BOARD_HAS_ON_RESET 1

#define PX4_GPIO_INIT_LIST { \
		GPIO_TONE_ALARM_IDLE,             \
		GPIO_SBUS_INV,                    \
	}

#define BOARD_ENABLE_CONSOLE_BUFFER
#define BOARD_CONSOLE_BUFFER_SIZE (1024*3)

/* I2C bus clock init - 1 bus (I2C1) */
#define BOARD_I2C_BUS_CLOCK_INIT {100000}

#define FLASH_BASED_PARAMS

__BEGIN_DECLS

#ifndef __ASSEMBLY__

extern void stm32_spiinitialize(void);
extern void stm32_usbinitialize(void);
extern void board_peripheral_reset(int ms);

#include <px4_platform_common/board_common.h>

#endif /* __ASSEMBLY__ */

__END_DECLS
