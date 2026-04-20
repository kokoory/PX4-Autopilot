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
- `Tools/simulation/gz_models_singlecopter/singlecopter/` — Gazebo SDF with lift/drag plugins
  - NOT inside the `gz` submodule (upstream only). Added to `GZ_SIM_RESOURCE_PATH`
    via `PX4_GZ_MODELS_EXTRA` in `src/modules/simulation/gz_bridge/gz_env.sh.in`

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

## Initial Setup (fresh clone)
```bash
git clone https://github.com/kokoory/PX4-Autopilot.git
cd PX4-Autopilot
git checkout claude/knowledge-distillation-flight-fGsy1
git submodule update --init --recursive
bash Tools/setup/ubuntu.sh           # installs build deps + Gazebo

# Distillation pipeline venv (separate from PX4 build env)
cd Tools/distillation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # core deps only
# Optional: pip install tensorflow tf2onnx   # for TFLite export
deactivate

# PX4 build (use system python, NOT venv)
cd ~/PX4-Autopilot
make px4_sitl gz_x500                # baseline quad test
make px4_sitl_neural gz_singlecopter # our singlecopter + neural control
make holybro_kakutef4aio_default     # Kakute F4 AIO firmware
```

## Troubleshooting

### Submodule fetch failures
- `Tools/simulation/gz` submodule fetch error for commit 468d164:
  Fixed in commit 2f33cd9. Pull latest and re-init submodules.
- Empty heatshrink/gps-devices dir: `git submodule deinit -f <path>`
  then `rm -rf <path> && git submodule update --init <path>`

### Python 3.13 + tflite-runtime
tflite-runtime does not support Python 3.13. Core training works without it.
Install `tensorflow` (includes tflite) separately if TFLite export is needed.

### Build env vs distillation env
Don't use the distillation venv for `make px4_sitl_*`. The venv lacks
kconfiglib/jinja2. Either `deactivate` or use a new terminal.

### Flash overflow on Kakute F4 AIO
F405 is 1MB flash. TFLite Micro is 274KB. If build exceeds flash, trim
modules in `default.px4board` (EKF2, GPS, DShot, OSD, TempComp already removed).

### mc_nn_control output wiring
- `MC_NN_NUM_MOT=1`, `MC_NN_NUM_SRV=4` for singlecopter (default)
- `MC_NN_NUM_MOT=4`, `MC_NN_NUM_SRV=0` for quadrotor
- NN output layout: `[motor_0..motor_N, servo_0..servo_M]`
- Motor outputs → actuator_motors (RPM scaling via MC_NN_MAX_RPM)
- Servo outputs → actuator_servos (direct [-1, 1] deflection)

## Key files quick reference
| File | Purpose |
|------|---------|
| `src/modules/mc_nn_control/mc_nn_control.cpp:Run()` | Main 400Hz control loop with safety layer |
| `src/modules/mc_nn_control/distillation_monitor.hpp` | Output validation, PID fallback trigger |
| `Tools/distillation/feedback_loop.py` | CLI entry for init/feedback/export cycles |
| `Tools/distillation/sim_real_bridge.py` | LLM sim-to-real gap reasoning (core novelty) |
| `Tools/distillation/flight_analyzer.py` | ULog parser → anomaly + phase stats → LLM prompt |
| `msg/DistillationStatus.msg` | uORB telemetry (incl. motor_saturation, flight_phase) |
| `ROMFS/px4fmu_common/init.d-posix/airframes/4030_gz_singlecopter` | Airframe config w/ vane mixing |
| `Tools/simulation/gz_models_singlecopter/singlecopter/model.sdf` | Gazebo physics model |
| `boards/holybro/kakutef4aio/` | F4 AIO board support (LED→motor repurpose) |

## Differentiation from Eureka (NVIDIA 2023)
Eureka: sim-only, offline, LLM generates reward once → RL → deploy → done.
Ours: closed-loop, LLM reasons about REAL flight data, iteratively refines
reward to bridge sim-to-real gap. Flight data includes per-phase stats,
anomaly detection, motor saturation, output rate of change — structured
input the LLM can physically interpret (motor lag, prop wash, vibration,
battery sag).
