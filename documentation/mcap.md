# MCAP format

`maus-export mcap` writes one file per run, `data/mcap/run_<id>.mcap`, using the **ROS 2 profile**
(schema encoding `ros2msg`, message encoding `cdr`, zstd chunks). The files open in
[Foxglove](https://foxglove.dev), `ros2 bag play`, and any MCAP reader.

## Topics

| Topic | Type | Content |
|---|---|---|
| `/maus/run_summary` | `maus_msgs/msg/RunSummary` | one message at run start: ids, robot, topping, test flag, order id, terminal status, success labels (−1 = unlabelled), duration, sensors, video URL |
| `/rosout` | `rcl_interfaces/msg/Log` | every machine log line (levels 10/20/30/40/50) |
| `/maus/phase` | `maus_msgs/msg/Phase` | phase transitions at `tsStart + elapsed_sec` |
| `/maus/vision_check` | `maus_msgs/msg/VisionCheck` | each vision check: phase, ok, message, errors[], image names |
| `/maus/camera/check/compressed` | `sensor_msgs/msg/CompressedImage` | raw check frames (JPEG), present only with `fetch --media` |
| `/maus/camera/check_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | annotated check overlays (JPEG) |
| `/maus/env/temperature` | `sensor_msgs/msg/Temperature` | ambient temperature at start (°C) |
| `/maus/env/humidity` | `sensor_msgs/msg/RelativeHumidity` | ambient humidity at start (0–1) |
| `/maus/hotplate/setpoint` | `sensor_msgs/msg/Temperature` | hotplate setpoint (°C) |

The custom `maus_msgs` definitions are in [`code/maus_export/msgdefs.py`](../code/maus_export/msgdefs.py)
and are embedded in each file, so no ROS install is needed to read them.

**Metadata records:**
- `run`: ids, robot, status, UTC start/end, the measured robot-clock offset, video URL, and prev/next run.
- `parameters`: all ~150 recipe/machine parameters (non-strings are JSON-encoded).

**Attachment:** `video.mp4` is embedded only with `fetch --video` followed by `mcap --video`.

All timestamps are UTC nanoseconds. Robot-local log times are shifted by the offset measured between the
first log line and `tsStart`.

## Reading in Python
```python
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

with open("data/mcap/run_3348.mcap", "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros in reader.iter_decoded_messages(topics=["/maus/vision_check"]):
        print(message.log_time, ros.phase, ros.ok, list(ros.errors))
```

## Viewing
Open the file in Foxglove (app or web) and add these panels: **Log** (`/rosout`), **Image**
(`/maus/camera/check_overlay/compressed`), **Raw Messages** (`/maus/vision_check`, `/maus/phase`), and **State
Transitions** on `/maus/phase.phase`.
