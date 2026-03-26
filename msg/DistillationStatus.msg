# Distillation model status for knowledge-distilled neural network controllers
#
# Reports the health and performance of the distilled MLP running in mc_nn_control.
# Used for monitoring model inference quality, PID fallback state, and
# runtime feedback loop for sim-to-real distillation refinement.
#
# Publisher: mc_nn_control
# Subscriber: logger, mavlink (for ground station telemetry and feedback loop)

uint64 timestamp		# [us] Time since system start

uint8 model_id			# Active distilled model identifier
uint8 model_version		# Model version number

int32 inference_time_us		# [us] Last inference duration
int32 controller_time_us	# [us] Full controller loop duration

float32 output_variance		# Variance of motor outputs (stability indicator)
float32 mean_position_error	# [m] Running average position error

bool fallback_active		# True if PID fallback is engaged
uint8 fallback_reason		# Reason for fallback (see constants below)

uint32 total_inferences		# Total inference count since boot
uint32 failed_inferences	# Failed inference count (NaN, out of range, timeout)
float32 success_rate		# Inference success rate (0.0 to 1.0)

# Runtime feedback loop diagnostics (for sim-to-real gap analysis)
float32[4] motor_outputs	# Last motor command outputs [-1, 1]
float32[3] position_error_ned	# [m] Current position error in NED frame
float32[3] angular_velocity_raw	# [rad/s] Raw angular velocity for vibration analysis
float32 motor_saturation_ratio	# Fraction of motors at output limit [0, 1]
float32 output_rate_of_change	# Motor output change rate (smoothness indicator)
uint8 flight_phase		# Current flight phase (see constants below)

# Fallback reason constants
uint8 FALLBACK_NONE = 0
uint8 FALLBACK_OUTPUT_NAN = 1
uint8 FALLBACK_OUTPUT_RANGE = 2
uint8 FALLBACK_INFERENCE_TIMEOUT = 3
uint8 FALLBACK_CONSECUTIVE_ERRORS = 4
uint8 FALLBACK_MANUAL = 5

# Flight phase constants (for context-aware feedback analysis)
uint8 PHASE_IDLE = 0
uint8 PHASE_TAKEOFF = 1
uint8 PHASE_HOVER = 2
uint8 PHASE_CRUISE = 3
uint8 PHASE_MANEUVER = 4
uint8 PHASE_LANDING = 5
