"""Offline test: synthetic crepe_get-shaped run -> MCAP -> read back."""

import json
from collections import Counter
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from maus_export.to_mcap import run_to_mcap
from maus_export.to_table import run_row

RUN = {
    "id": 1,
    "machineId": 3,
    "machineName": "maus-test",
    "isTest": True,
    "topping": "plain",
    "orderId": "",
    "tsStart": "2026-10-30T16:45:37.000Z",
    "tsEnd": "2026-10-30T16:49:19.000Z",
    "hotplateTemp": 190,
    "envTempStart": 45.6,
    "envHumidityStart": 19.17,
    "successSpread": None,
    "successTopping": None,
    "successFold": None,
    "successHandover": None,
    "comments": None,
    "batch": None,
    "terminal_status": "succeeded",
    "prevId": None,
    "nextId": 2,
    "videoUrl": "https://example.invalid/1.mp4",
    "logs": [  # robot-local time (Zurich, UTC+1 in late October)
        {
            "msg": "Executing make_crepe action",
            "name": "make_crepe_action",
            "level": 20,
            "timestamp": "2026-10-30T17:45:37.514",
        },
        {
            "msg": "hotplate demand is ON but duet reports the SSR OFF",
            "name": "hotplate_controller",
            "level": 40,
            "timestamp": "2026-10-30T17:45:58.757",
        },
        {
            "msg": "fold.spatula_gate_enabled is false",
            "name": "make_crepe_action",
            "level": 30,
            "timestamp": "2026-10-30T17:47:15.177",
        },
    ],
    "phases": [
        {
            "phase": "init",
            "detail": "",
            "progress": 0,
            "timestamp": "2026-10-30T17:45:37",
            "elapsed_sec": 0.009,
        },
        {
            "phase": "spread_batter_concentric",
            "detail": "",
            "progress": 0.2,
            "timestamp": "2026-10-30T17:45:54",
            "elapsed_sec": 17.394,
        },
        {
            "phase": "cook",
            "detail": "",
            "progress": 0.5,
            "timestamp": "2026-10-30T17:46:21",
            "elapsed_sec": 43.644,
        },
    ],
    "visionChecks": [
        {
            "ok": True,
            "frame": "00_spread_frame.jpg",
            "overlay": "00_spread_overlay.jpg",
            "phase": "spread",
            "errors": [],
            "message": "crepe spread successfully",
            "timestamp": "2026-10-30T17:46:21",
        },
        {
            "ok": False,
            "frame": "01_push_off_frame.jpg",
            "overlay": "01_push_off_overlay.jpg",
            "phase": "push_off",
            "errors": ["crepe not aligned with the plate"],
            "message": "not ready for push off",
            "timestamp": "2026-10-30T17:47:52",
        },
    ],
    "parameters": {
        "dispense.batter.total_cook_time": 54,
        "fold.spatula_gate_enabled": False,
        "machine.pos_idle": "G1 X0",
    },
}


def test_roundtrip(tmp_path: Path):
    media = tmp_path / "media" / "1"
    media.mkdir(parents=True)
    (media / "00_spread_frame.jpg").write_bytes(b"\xff\xd8fakejpeg")
    out = tmp_path / "run_1.mcap"
    run_to_mcap(RUN, out, tmp_path / "media")

    with open(out, "rb") as f:
        msgs = [
            (ch.topic, m.log_time, ros)
            for _, ch, m, ros in make_reader(
                f, decoder_factories=[DecoderFactory()]
            ).iter_decoded_messages()
        ]
    counts = Counter(topic for topic, _, _ in msgs)
    assert counts["/rosout"] == 3
    assert counts["/maus/phase"] == 3
    assert counts["/maus/vision_check"] == 2
    assert counts["/maus/camera/check/compressed"] == 1
    assert counts["/maus/run_summary"] == 1

    rosout = [(t, ros) for topic, t, ros in msgs if topic == "/rosout"]
    # first log at 17:45:37.514 local == 16:45:37.514 UTC
    assert rosout[0][0] == 1793378737514000000
    assert rosout[1][1].level == 40

    vc = [ros for topic, _, ros in msgs if topic == "/maus/vision_check"]
    assert vc[1].ok is False and vc[1].errors == [
        "crepe not aligned with the plate"
    ]

    with open(out, "rb") as f:
        meta = {m.name: m.metadata for m in make_reader(f).iter_metadata()}
    assert meta["run"]["local_clock_offset_min"] == "60"
    assert (
        json.loads(meta["parameters"]["dispense.batter.total_cook_time"]) == 54
    )


def test_table_row():
    row = run_row(RUN)
    assert row["status"] == "succeeded"
    assert row["first_failed_check"] == "push_off"
    assert row["n_log_error"] == 1 and row["n_log_warn"] == 1
    assert abs(row["phase_s.cook"] - (222 - 43.644)) < 1e-6
