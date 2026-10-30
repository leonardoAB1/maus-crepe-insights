"""ROS 2 message definitions (ros2msg text) used in the MCAP files.

Standard types are copied from the ROS 2 interfaces so no ROS install is needed.
Dependencies are appended in the concatenated format that rosbag2/MCAP expects.
"""

SEP = "=" * 80

TIME = """int32 sec
uint32 nanosec
"""

HEADER = """builtin_interfaces/Time stamp
string frame_id
"""

LOG = """byte DEBUG=10
byte INFO=20
byte WARN=30
byte ERROR=40
byte FATAL=50
builtin_interfaces/Time stamp
uint8 level
string name
string msg
string file
string function
uint32 line
"""

COMPRESSED_IMAGE = """std_msgs/Header header
string format
uint8[] data
"""

TEMPERATURE = """std_msgs/Header header
float64 temperature
float64 variance
"""

RELATIVE_HUMIDITY = """std_msgs/Header header
float64 relative_humidity
float64 variance
"""

# --- custom maus_msgs -------------------------------------------------------

PHASE = """# One step of the make_crepe action (init, oil, spread_batter_concentric, cook, ...)
builtin_interfaces/Time stamp
string robot
int64 run_id
string phase
string detail
float32 progress
float64 elapsed_sec
"""

VISION_CHECK = """# Result of one on-robot vision check. header.frame_id = robot name
std_msgs/Header header
int64 run_id
uint32 index
string phase
bool ok
string message
string[] errors
string frame_image
string overlay_image
"""

RUN_SUMMARY = """# One crepe run. success_* fields: 1 = ok, 0 = failed, -1 = not labelled
std_msgs/Header header
int64 run_id
int32 machine_id
string robot
string topping
bool is_test
string order_id
string terminal_status
int8 success_spread
int8 success_topping
int8 success_fold
int8 success_handover
float64 duration_sec
float64 hotplate_temp
float64 env_temp_start
float64 env_humidity_start
string comments
string video_url
"""


def _with_deps(main: str, deps: list[tuple[str, str]]) -> str:
    parts = [main.rstrip("\n")]
    for name, text in deps:
        parts.append(f"{SEP}\nMSG: {name}\n{text.rstrip(chr(10))}")
    return "\n".join(parts) + "\n"


_TIME_DEP = ("builtin_interfaces/Time", TIME)
_HEADER_DEPS = [("std_msgs/Header", HEADER), _TIME_DEP]

# datatype -> full msgdef text
SCHEMAS: dict[str, str] = {
    "rcl_interfaces/msg/Log": _with_deps(LOG, [_TIME_DEP]),
    "sensor_msgs/msg/CompressedImage": _with_deps(COMPRESSED_IMAGE, _HEADER_DEPS),
    "sensor_msgs/msg/Temperature": _with_deps(TEMPERATURE, _HEADER_DEPS),
    "sensor_msgs/msg/RelativeHumidity": _with_deps(RELATIVE_HUMIDITY, _HEADER_DEPS),
    "maus_msgs/msg/Phase": _with_deps(PHASE, [_TIME_DEP]),
    "maus_msgs/msg/VisionCheck": _with_deps(VISION_CHECK, _HEADER_DEPS),
    "maus_msgs/msg/RunSummary": _with_deps(RUN_SUMMARY, _HEADER_DEPS),
}
