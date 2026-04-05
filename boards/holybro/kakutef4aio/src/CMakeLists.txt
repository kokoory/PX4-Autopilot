############################################################################
#
#   Copyright (c) 2024 PX4 Development Team. All rights reserved.
#
############################################################################

add_library(drivers_board
	i2c.cpp
	init.c
	spi.cpp
	timer_config.cpp
)

target_link_libraries(drivers_board
	PRIVATE
		arch_spi
		nuttx_arch
		nuttx_drivers
		px4_layer
)
