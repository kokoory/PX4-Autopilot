# Distillation model status for knowledge-distilled neural network controllers
#
# Reports the health and performance of the distilled MLP running in mc_nn_control.
# Used for monitoring model inference quality and PID fallback state.
#
# Publisher: mc_nn_control
# Subscriber: logger, mavlink (for ground station telemetry)

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

# Fallback reason constants
uint8 FALLBACK_NONE = 0
uint8 FALLBACK_OUTPUT_NAN = 1
uint8 FALLBACK_OUTPUT_RANGE = 2
uint8 FALLBACK_INFERENCE_TIMEOUT = 3
uint8 FALLBACK_CONSECUTIVE_ERRORS = 4
uint8 FALLBACK_MANUAL = 5
