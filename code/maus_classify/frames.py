"""One row per vision check attempt (frame) with weak labels and a split."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from maus_export.to_mcap import Clock

from .split import SPLIT_VERSION, TEST, robot_split

OUTPUT = Path("training") / "frames.parquet"
RUNS_DIR = Path("raw") / "runs"
MEDIA_DIR = Path("raw") / "media"
# A passed check proves the crepe has this shape at that point
FOLD_CLASS = {"spread": "flat", "fold1": "half", "fold2": "quarter"}
REASON_SEP = "; "
_SPACES = re.compile(r"\s+")

SCHEMA = pa.schema(
    [
        ("run_id", pa.int64()),
        ("check", pa.string()),
        ("check_index", pa.int32()),
        ("attempt", pa.int32()),
        ("ts_utc", pa.timestamp("us", tz="UTC")),
        ("robot", pa.string()),
        ("ok", pa.bool_()),
        ("message", pa.string()),
        ("errors", pa.list_(pa.string())),
        ("frame_path", pa.string()),
        ("is_last_attempt", pa.bool_()),
        ("fold_class", pa.string()),
        ("reason", pa.string()),
        ("split", pa.string()),
        ("split_version", pa.string()),
    ]
)


class FramesError(ValueError):
    """Input problem with a fixed message that is safe to print."""


def normalize(text: str) -> str:
    """Lowercase, trim and collapse whitespace."""
    return _SPACES.sub(" ", text).strip().lower()


def reason(ok: bool | None, errors: list[str], message: str | None) -> Any:
    """Normalized failure reason; null when the check passed."""
    if ok:
        return None
    parts = sorted({normalize(e) for e in errors if e and normalize(e)})
    if parts:
        return REASON_SEP.join(parts)
    if message and normalize(message):
        return normalize(message)
    return None


def fold_class(check: str | None, ok: bool | None) -> str | None:
    """Shape label from a passed spread or fold check, else null."""
    return FOLD_CLASS.get(check or "") if ok else None


def run_frames(run: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    """Frame rows for one raw `crepe_get` run, without the split."""
    checks = run.get("visionChecks") or []
    clock = Clock(run)
    last: dict[str, int] = {}
    for i, vc in enumerate(checks):
        last[vc.get("phase")] = i
    seen: dict[str, int] = {}
    rows = []
    for i, vc in enumerate(checks):
        name = vc.get("phase")
        seen[name] = seen.get(name, 0) + 1
        ok = vc.get("ok")
        errors = [str(e) for e in vc.get("errors") or []]
        frame = vc.get("frame")
        rel = MEDIA_DIR / str(run["id"]) / frame if frame else None
        ts = vc.get("timestamp")
        rows.append(
            {
                "run_id": run["id"],
                "check": name,
                "check_index": i,
                "attempt": seen[name],
                "ts_utc": clock.local(ts) if ts else None,
                "robot": run.get("machineName"),
                "ok": ok,
                "message": vc.get("message"),
                "errors": errors,
                "frame_path": rel.as_posix()
                if rel and (data_dir / rel).is_file()
                else None,
                "is_last_attempt": last[name] == i,
                "fold_class": fold_class(name, ok),
                "reason": reason(ok, errors, vc.get("message")),
            }
        )
    return rows


def runs_to_frames(runs: list[dict[str, Any]], data_dir: Path) -> pd.DataFrame:
    """Raw runs -> frame table sorted by run id and check order."""
    ids = [r["id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise FramesError("duplicate run ids in raw runs")
    # The cutoff is set over runs that have checks, whatever their verdict,
    # so older runs from before images were recorded do not shift it
    with_checks = [r for r in runs if r.get("visionChecks")]
    index = [r["id"] for r in with_checks]
    starts = pd.Series(
        pd.to_datetime([r["tsStart"] for r in with_checks], utc=True),
        index=index,
        dtype="datetime64[ns, UTC]",
    )
    robots = pd.Series(
        [r.get("machineName") for r in with_checks], index=index, dtype=object
    )
    split = robot_split(starts, robots)
    rows = []
    for run in sorted(runs, key=lambda r: r["id"]):
        for row in run_frames(run, data_dir):
            row["split"] = split[run["id"]]
            row["split_version"] = SPLIT_VERSION
            rows.append(row)
    df = pd.DataFrame(rows, columns=SCHEMA.names)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _write_atomic(table: pa.Table, output: Path) -> None:
    """Swap in a finished file, so a failed write keeps the old output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".frames-", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_frames(data_dir: Path) -> pd.DataFrame:
    """Build `training/frames.parquet` from the raw runs in `data_dir`."""
    files = sorted((data_dir / RUNS_DIR).glob("*.json"))
    if not files:
        raise FramesError(
            f"no raw run JSON files in {data_dir / RUNS_DIR}; fetch first"
        )
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    df = runs_to_frames(runs, data_dir)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    output = data_dir / OUTPUT
    _write_atomic(table, output)
    n_test = int((df["split"] == TEST).sum())
    print(
        f"{len(runs)} runs, {len(df)} frames ({n_test} test, "
        f"{int(df['frame_path'].isna().sum())} without image) -> {output}"
    )
    return df
