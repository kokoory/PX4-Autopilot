uint64 timestamp    # time since system start (microseconds)
float32 angle_rad   # rotor angular position in radians (0 to 2*PI)
float32 angle_raw   # raw encoder value (0-16383 for 14-bit)
float32 rpm         # current RPM
bool valid          # true if encoder data is valid
