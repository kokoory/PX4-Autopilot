#include <px4_arch/spi_hw_description.h>

constexpr px4_spi_bus_t px4_spi_buses[SPI_BUS_MAX_BUS_ITEMS] = {
	initSPIBus(SPI::Bus::SPI1, {
		initSPIDevice(DRV_IMU_DEVTYPE_ICM20689, SPI::CS{GPIO::PortC, GPIO::Pin4}, SPI::DRDY{GPIO::PortC, GPIO::Pin5}),
	}),
	initSPIBus(SPI::Bus::SPI3, {
		initSPIDevice(DRV_OSD_DEVTYPE_ATXXXX, SPI::CS{GPIO::PortB, GPIO::Pin14}),
		initSPIDevice(DRV_FLASH_DEVTYPE_JEDEC, SPI::CS{GPIO::PortB, GPIO::Pin3}),
	}),
};
