# CLAUDE.md - PX4-Autopilot

## Role

You are a UAV (Unmanned Aerial Vehicle) systems expert specializing in flight controller firmware, real-time embedded systems, and autonomous flight software. You have deep knowledge of PX4 autopilot architecture, MAVLink protocol, sensor fusion (EKF2), flight control theory, and safety-critical software development.

## Project Overview

PX4-Autopilot is an open-source autopilot firmware for drones and other unmanned vehicles. It runs on NuttX RTOS and POSIX platforms, supporting multicopters, fixed-wing, VTOL, rovers, and more. This is **safety-critical software** — incorrect changes can cause crashes and property damage.

## Branch: `claude/create-claude-profile-tCSLg`

### What this branch adds

#### Swashplateless Helicopter Control
Sinusoidal motor speed modulation synchronized with rotor angular position to achieve
cyclic pitch/roll control without a mechanical swashplate. Based on the algorithm from
[jonahdeclerck/swashplateless_helicopter](https://github.com/jonahdeclerck/swashplateless_helicopter).

**Core mixing formula** (`ActuatorEffectivenessHelicopterSwashplateless::updateSetpoint()`):
```
main_motor = throttle + pitch_cmd * cos(rotor_angle + phase)
                      + roll_cmd  * sin(rotor_angle + phase)

tail_motor = yaw_cmd * yaw_sign + throttle * yaw_throttle_scale
```

When the blade is at an angular position where more lift is needed, the motor speeds up;
at the opposite position, it slows down. This cyclically varies lift around the rotation,
tilting the rotor disc — exactly what a swashplate does mechanically.

**Files:**

| Path | Purpose |
|------|---------|
| `src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessHelicopterSwashplateless.hpp` | Class definition: inherits `ModuleParams` + `ActuatorEffectiveness`, subscribes to `rotor_position` uORB topic |
| `src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessHelicopterSwashplateless.cpp` | Core implementation: `updateSetpoint()` with sinusoidal mixing, throttle curve interpolation, spoolup logic, saturation reporting |
| `src/modules/control_allocator/ControlAllocator.hpp` | Added `HELICOPTER_SWASHPLATELESS = 16` to `EffectivenessSource` enum |
| `src/modules/control_allocator/ControlAllocator.cpp` | Added switch case to instantiate `ActuatorEffectivenessHelicopterSwashplateless` |
| `src/modules/control_allocator/module.yaml` | Added `CA_AIRFRAME` value 16, mixer config (2 motors), and `CA_SWL_PHASE` parameter definition |
| `ROMFS/px4fmu_common/init.d/airframes/16002_helicopter_swashplateless` | Airframe: `CA_AIRFRAME=16`, throttle curve, yaw config, AS5047 enable, PWM mapping |
| `msg/RotorPosition.msg` | uORB message: `angle_rad` (float32), `angle_raw` (float32), `rpm` (float32), `valid` (bool) |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `CA_AIRFRAME` | enum | 16 | Helicopter (Swashplateless) |
| `CA_SWL_PHASE` | float | 0.0 | Phase offset for blade lag compensation (degrees, -180 to 180) |
| `CA_HELI_THR_C0..C4` | float | 0.0..1.0 | 5-point throttle curve |
| `CA_HELI_YAW_CCW` | bool | 0 | Main rotor turns counter-clockwise |
| `CA_HELI_YAW_TH_S` | float | 0.1 | Yaw compensation scale based on throttle |
| `COM_SPOOLUP_TIME` | float | 10.0 | Throttle ramp-up time on arming (seconds) |
| `SENS_EN_AS5047` | enum | 1 | Enable AS5047 magnetic encoder |

**Control flow:**
```
mc_att_control (attitude controller)
    → roll_cmd, pitch_cmd, yaw_cmd, thrust_z
    ↓
ControlAllocator (CA_AIRFRAME=16)
    → ActuatorEffectivenessHelicopterSwashplateless::updateSetpoint()
    ↓
    reads rotor_position uORB (angle_rad from AS5047)
    ↓
    main_motor = constrain(throttle + cyclic_modulation, 0, 1)
    tail_motor = yaw * yaw_sign + throttle * yaw_throttle_scale
    ↓
actuator_motors → PWM driver
    CH1: Main motor ESC
    CH2: Tail motor ESC
```

**Known limitations (identified during verification):**
- `mainMotorEnaged()` typo (matches upstream helicopter code, not introduced here)
- Roll/pitch saturation flags never set (cyclic is via motor speed, not direct actuators)
- `spoolup_time=0` causes divide-by-zero (same as upstream helicopter code)

#### Magnetic Encoder RPM Drivers

Two new drivers for ams-OSRAM 14-bit magnetic rotary encoders:

**AS5047 (SPI)** — `src/drivers/rpm/as5047/`

| File | Lines | Purpose |
|------|-------|---------|
| `AS5047.hpp` | 130 | Class def: `device::SPI` + `I2CSPIDriver<AS5047>`, SPI Mode 1, parity calculation |
| `AS5047.cpp` | 286 | Register read (2-frame SPI protocol), angle→RPM calculation, EMA filter (α=0.1), publishes `rpm` + `rotor_position` |
| `as5047_main.cpp` | 79 | CLI: `as5047 start -s -b <bus>` |
| `parameters.yaml` | 40 | `SENS_EN_AS5047`, `AS5047_POLL` (default 1000us), `AS5047_POLES` (default 1) |

- SPI protocol: send read command with parity → NOP → read response, check error flag + parity
- Angle register: `0x3FFF` (ANGLECOM), 14-bit (0-16383 = 0°-360°)
- RPM: `(delta_angle / 16384) * (1e6 / delta_time_us) * 60 / pole_pairs`
- Also publishes `rotor_position` uORB topic (angle_rad, angle_raw, rpm, valid)
- Device type: `DRV_SENS_DEVTYPE_AS5047` (0xF2)

**AS5048B (I2C)** — `src/drivers/rpm/as5048b/`

| File | Lines | Purpose |
|------|-------|---------|
| `AS5048B.hpp` | 104 | Class def: `device::I2C` + `I2CSPIDriver<AS5048B>`, registers 0xFE/0xFF |
| `AS5048B.cpp` | 202 | I2C register read, angle combine `(high<<6) | (low & 0x3F)`, EMA filter, publishes `rpm` |
| `as5048b_main.cpp` | 82 | CLI: `as5048b start -a 0x40` |
| `parameters.yaml` | 38 | `SENS_EN_AS5048B`, `AS5048B_POLL`, `AS5048B_POLES` |

- Default I2C address: 0x40 (configurable A1/A2 pins → 0x40-0x43)
- Angle: ANGLE_HIGH (0xFE, bits 13:6) + ANGLE_LOW (0xFF, bits 5:0)
- Device type: `DRV_SENS_DEVTYPE_AS5048B` (0xF1)

**Usage:**
```bash
# Board defconfig
CONFIG_DRIVERS_RPM_AS5047=y    # SPI version
CONFIG_DRIVERS_RPM_AS5048B=y   # I2C version

# NuttX console
as5047 start -s -b 1           # SPI bus 1
as5048b start -a 0x40          # I2C address 0x40
as5047 status                  # Check RPM readings

# QGC parameters
SENS_EN_AS5047 = 1             # Auto-start
AS5047_POLES = 7               # 14-pole motor = 7 pole pairs
```

#### HITL Airframe
- `ROMFS/px4fmu_common/init.d/airframes/1003_singlecopter.hil` — HIL singlecopter (1 motor + 4 vanes)

## Architecture

- **uORB**: Inter-module publish/subscribe messaging system. Message definitions in `msg/`.
- **Modules** (`src/modules/`): Independent processes communicating via uORB topics.
  - `commander` — Vehicle state machine and mode management
  - `navigator` — Mission execution, RTL, Land, and other autonomous modes
  - `ekf2` — Extended Kalman Filter for state estimation (position, velocity, attitude)
  - `mc_att_control` / `mc_pos_control` — Multicopter attitude and position controllers
  - `fw_att_control` / `fw_pos_control` — Fixed-wing controllers
  - `vtol_att_control` — VTOL transition and mode management
  - `control_allocator` — Actuator mixing (CA_AIRFRAME selection, effectiveness matrices)
  - `mavlink` — MAVLink protocol communication
  - `sensors` — Sensor data processing and calibration
  - `logger` — Flight data logging (ULog format)
- **Drivers** (`src/drivers/`): Hardware abstraction for IMU, GPS, barometer, magnetometer, RPM encoders, etc.
- **Libraries** (`src/lib/`): Reusable math, control, and utility libraries.
- **Platforms** (`platforms/`): NuttX, POSIX, QuRT, ROS2 platform abstraction layers.
- **Boards** (`boards/`): Board-specific configurations for 45+ hardware targets.

## Build & Test

```bash
# SITL (Software-In-The-Loop) build
make px4_sitl_default

# Specific board target
make px4_fmu-v6x_default

# Unit tests (GTest)
make tests

# Selective tests
TESTFILTER=<pattern> make tests

# Integration tests (MAVSDK)
make tests_integration

# Code formatting check (astyle)
make check_format

# Static analysis (clang-tidy)
make clang-tidy

# Test coverage
make tests_coverage
```

## Code Standards

- **C++17** (CMAKE_CXX_STANDARD 17), **C11** for C code
- **astyle** formatting enforced — run `make check_format` before committing
- **clang-tidy** static analysis enforced in CI
- **Python**: mypy + flake8
- Follow existing code style in surrounding files

## Commit Convention

PX4 uses [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

**Scopes**: Use the module/subsystem name — `ekf2`, `commander`, `navigator`, `mavlink`, `mc_att_control`, `fw_att_control`, `vtol`, `sensors`, `drivers`, `boards/px4_fmu-v6x`, `simulation`, `uorb`, `param`, `logger`, `battery`, `actuators`, `ci`, `build`, `docs`, `helicopter`

**Examples**:
```
feat(ekf2): add height fusion timeout
fix(mavlink): correct BATTERY_STATUS_V2 parsing
refactor(navigator): simplify RTL altitude logic
feat(helicopter): add swashplateless control using rotor angle modulation
feat(drivers): add AS5047 SPI magnetic encoder RPM driver
```

## Key Domain Knowledge

### Flight Control
- PID/cascaded control loops: outer (position) -> inner (velocity) -> innermost (attitude/rate)
- Control loops run at fixed rates (typically 250Hz for attitude, 50Hz for position)
- All control math uses NED (North-East-Down) coordinate frame

### Control Allocation (Actuator Mixing)
- `CA_AIRFRAME` parameter selects vehicle type (0=multirotor, 10-12=helicopter, 16=swashplateless)
- `ActuatorEffectiveness` classes define how control axes map to actuators
- `updateSetpoint()` is called every cycle — this is where mixing happens
- For helicopters: throttle curve + swashplate servo mixing (or sinusoidal motor modulation for swashplateless)

### State Estimation (EKF2)
- Fuses IMU, GPS, barometer, magnetometer, optical flow, vision
- Critical for safe flight — changes require extensive testing with real flight logs
- Tuning parameters accessible via `param` system

### MAVLink Protocol
- Standard protocol for GCS-vehicle communication
- Message definitions are auto-generated from XML schemas
- Handles telemetry, commands, mission upload/download, parameter sync

### uORB Messaging
- Topic definitions: `msg/*.msg` files (IDL-like format)
- Auto-generated C++ headers from message definitions
- Zero-copy publish/subscribe within a single process

### Safety Considerations
- Always consider failsafe behavior when modifying flight-critical code
- Parameter changes can affect flight behavior — validate ranges and defaults
- Test with SITL before any hardware deployment
- Upload flight logs to https://logs.px4.io for review

## Key Files

| Path | Purpose |
|------|---------|
| `src/modules/commander/` | Vehicle state machine, arming, mode transitions |
| `src/modules/ekf2/` | State estimation (EKF) |
| `src/modules/navigator/` | Autonomous navigation modes |
| `src/modules/mc_pos_control/` | Multicopter position controller |
| `src/modules/control_allocator/` | Actuator mixing and allocation |
| `src/modules/mavlink/` | MAVLink protocol handler |
| `src/drivers/rpm/as5047/` | AS5047 SPI magnetic encoder driver |
| `src/drivers/rpm/as5048b/` | AS5048B I2C magnetic encoder driver |
| `msg/` | uORB message definitions |
| `msg/RotorPosition.msg` | Rotor angle/RPM from magnetic encoder |
| `boards/` | Board-specific build configs |
| `ROMFS/px4fmu_common/` | Default parameters and startup scripts |
| `ROMFS/px4fmu_common/init.d/airframes/16002_helicopter_swashplateless` | Swashplateless helicopter airframe |
| `Tools/` | Development utilities |
