/****************************************************************************
 *
 *   Copyright (c) 2024 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

/**
 * @file init.c
 *
 * Kakute F4 AIO V2.1 specific early startup code.
 */

#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/tasks.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <debug.h>
#include <errno.h>
#include <syslog.h>

#include <nuttx/board.h>
#include <nuttx/spi/spi.h>
#include <nuttx/i2c/i2c_master.h>
#include <nuttx/analog/adc.h>
#include <nuttx/mm/gran.h>

#include <stm32.h>
#include "board_config.h"
#include <stm32_uart.h>

#include <arch/board/board.h>

#include <drivers/drv_hrt.h>
#include <drivers/drv_board_led.h>

#include <systemlib/px4_macros.h>

#include <px4_arch/io_timer.h>
#include <px4_platform_common/init.h>
#include <px4_platform/board_dma_alloc.h>

#if defined(FLASH_BASED_PARAMS)
#  include <parameters/flashparams/flashfs.h>
#endif

__BEGIN_DECLS
extern void led_init(void);
extern void led_on(int led);
extern void led_off(int led);
__END_DECLS

__EXPORT void board_peripheral_reset(int ms)
{
	UNUSED(ms);
}

__EXPORT void board_on_reset(int status)
{
	for (int i = 0; i < DIRECT_PWM_OUTPUT_CHANNELS; ++i) {
		px4_arch_configgpio(io_timer_channel_get_gpio_output(i));
	}

	if (status >= 0) {
		up_mdelay(400);
	}
}

__EXPORT void stm32_boardinitialize(void)
{
	board_on_reset(-1);

	/* configure LEDs */
	board_autoled_initialize();

	/* configure ADC pins */
	stm32_configgpio(GPIO_ADC1_IN13);	/* BATT_VOLTAGE_SENS PC3 */
	stm32_configgpio(GPIO_ADC1_IN12);	/* BATT_CURRENT_SENS PC2 */
	stm32_configgpio(GPIO_ADC1_IN11);	/* RSSI PC1 */

	/* configure SBUS inversion */
	stm32_configgpio(GPIO_SBUS_INV);

	/* configure SPI */
	stm32_spiinitialize();
}

__EXPORT int board_app_initialize(uintptr_t arg)
{
	px4_platform_init();

#if defined(FLASH_BASED_PARAMS)
	static sector_descriptor_t params_sector_map[] = {
		{1, 16 * 1024, 0x08004000},
		{0, 0, 0},
	};

	parameter_flashfs_init(params_sector_map, NULL, 0);
#endif

	/* Configure the DMA allocator */
	if (board_dma_alloc_init() < 0) {
		syslog(LOG_ERR, "[boot] DMA alloc init failed\n");
	}

	/* Configure the HW based on the manifest */
	px4_platform_configure();

	return OK;
}
