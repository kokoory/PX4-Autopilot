# Knowledge Distillation for Singlecopter Flight Control

## Branch: `claude/knowledge-distillation-flight-fGsy1`

## Project Summary
LLM-guided knowledge distillation pipeline for real-time neural network flight control.
Core novelty: **Runtime Distillation Feedback Loop** — real flight data feeds back to LLM,
which reasons about sim-to-real gap and generates improved reward functions.

## Target Vehicle
**Ball Drone MK II** — singlecopter (1 motor + 4 servo vanes)
- Layout (top→bottom): FC/Battery → Motor → Propeller → 4 Control Vanes → Landing legs
- CoG above thrust vectoring surfaces (pendulum stability)
- Hardware: Pixhawk 4 (STM32F765) + MD85MG-CAN servos (DroneCAN) + brushless motor

## What this branch adds

### Phase 2: Distillation Engine (`Tools/distillation/`)
- `reward_generator.py` — Eureka-style LLM reward function generation
- `train_student.py` — PyTorch MLP training (behavior cloning + RL)
- `export_tflite.py` — PyTorch → ONNX → TFLite → C++ byte array
- `validate_model.py` — Model size, accuracy, stability verification
- `scenarios.yaml` — Ball Drone vehicle params + flight scenarios
- `model_config.yaml` — MLP [15→64→64→32→5] (1 motor + 4 servos)

### Runtime Feedback Loop (core novelty, differentiates from Eureka)
- `flight_analyzer.py` — ULog parser, anomaly detection, per-phase statistics
- `sim_real_bridge.py` — LLM-based sim-to-real gap reasoning
- `feedback_loop.py` — Closed-loop orchestrator (init → feedback → export)

### Phase 3: PX4 Safety Integration (`src/modules/mc_nn_control/`)
- `distillation_monitor.hpp` — Output validation, timing check, auto PID fallback
- `DistillationStatus.msg` — uORB telemetry with feedback diagnostics
- Extended params: MC_NN_FALLBACK, MC_NN_MAX_INF_T, MC_NN_ERR_LIM, MC_NN_MODEL_ID

### Singlecopter Model
- `ROMFS/.../airframes/4030_gz_singlecopter` — PX4 airframe (CA_AIRFRAME=9, Custom)
- `Tools/simulation/gz/models/singlecopter/` — Gazebo SDF with lift/drag plugins

## Key Commands

```bash
# SITL simulation
make px4_sitl_neural gz_singlecopter

# Pixhawk 4 hardware build
make px4_fmu-v5_default

# Distillation pipeline
cd Tools/distillation
python feedback_loop.py init --scenario hover_stable
python feedback_loop.py feedback --flight-log <path_to.ulg>
python feedback_loop.py export
python feedback_loop.py status

# Individual steps
python reward_generator.py --scenario hover_stable
python train_student.py --mode behavior_cloning
python export_tflite.py --model output/student.pt
python validate_model.py --tflite output/student.tflite --pytorch output/student.pt
```

## Hardware Config (real flight)
- FC: **Kakute F4 AIO V2.1** (STM32F405, 168MHz, 1MB flash)
- Build: `make holybro_kakutef4aio_default`
- Bootloader: Custom PX4 bootloader (`~/PX4-Bootloader`, target `kakutef4aio_bl`)
  - Flash via DFU: `sudo dfu-util -a 0 -s 0x08000000:leave -D ~/PX4-Bootloader/build/kakutef4aio_bl/kakutef4aio_bl.bin`
  - Board type: 122 (AP_HW_KAKUTEF4), VBUS sense disabled
- Upload: `make holybro_kakutef4aio_default upload`
- Servos: PWM servo (later upgrade to MD85MG-CAN for CAN bus)
- All outputs PWM:
  - M1 (PB0, Timer3) → Servo V0 (Front)  PWM_MAIN_FUNC1=201
  - M2 (PB1, Timer3) → Servo V1 (Right)  PWM_MAIN_FUNC2=202
  - M3 (PA3, Timer5) → Servo V2 (Back)   PWM_MAIN_FUNC3=203
  - M4 (PA2, Timer5) → Servo V3 (Left)   PWM_MAIN_FUNC4=204
  - LED/M5 (PC8, Timer8) → Motor ESC     PWM_MAIN_FUNC5=101
- Motor: LED pad (PC8) repurposed as ESC output
- IMU: ICM20689 on SPI1 (CS=PC4, DRDY=PC5)
- Flash usage: ~87% (885KB/992KB) — TFLite is 274KB
- Modules removed for flash: EKF2, GPS, DShot, OSD, TempComp, HoverThrust, GyroCalib
- Safety: `MC_NN_FALLBACK = 1` (always!)

### Build fixes applied (kakutef4aio)
- `board_config.h`: Added `px4_config.h` and `stm32_gpio.h` includes
- `spi.cpp`: Added `drv_sensor.h`, removed non-existent `DRV_FLASH_DEVTYPE_JEDEC`
- `timer_config.cpp`: Full DMA stream/channel specs (TIM3:S2/C5, TIM5:S0/C6, TIM8:S1/C7)
- `defconfig`: Disabled NuttX TIM3/TIM5/TIM8 (PX4 IO timer conflict)
- `default.px4board`: Trimmed modules to fit TFLite in 992KB flash

## Architecture
```
LLM → reward_v1 → RL(sim) → MLP_v1 → deploy → fly
                                                 ↓
LLM → reward_v2 ← sim_real_bridge ← flight_analyzer ← ULog
       (improved)   (LLM reasons     (anomaly detection,
                     about gap)       phase statistics)
```
