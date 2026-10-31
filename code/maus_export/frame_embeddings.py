"""Cache frozen-CNN embeddings for every raw check frame.

data/raw/media/<run_id>/<NN>_<check>_frame.jpg
    -> data/training/frame_embeddings.parquet

One row per frame, keyed by (run_id, check, attempt) like the frame table,
with one float32 column per embedding dimension (e0..e{D-1}). Runs with
onnxruntime only (ml env); the model is code/models/frame-embedder/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .frame_preprocess import load_frame, preprocess

log = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "frame-embedder"
    / "model.onnx"
)
OUT_NAME = Path("training") / "frame_embeddings.parquet"
MEDIA_DIR = Path("raw") / "media"
RUNS_DIR = Path("raw") / "runs"
FRAME_RE = re.compile(r"^(?P<index>\d+)_(?P<check>.+)_frame\.jpg$")
EMBED_PREFIX = "e"
KEY = ["run_id", "check", "attempt"]
BATCH_SIZE = 32
HASH_CHUNK = 1 << 20


def scan_frames(data_dir: Path) -> pd.DataFrame:
    """One row per raw frame: run_id, check, attempt, check_index,
    frame_path (relative to data_dir, posix). attempt is the 1-based
    occurrence of the check within the run, in visionChecks order, and
    check_index the visionChecks index (NN), matching the frame table."""
    rows = []
    for path in sorted((data_dir / MEDIA_DIR).glob("*/*_frame.jpg")):
        m = FRAME_RE.match(path.name)
        if not m or not path.parent.name.isdigit():
            continue
        rows.append(
            {
                "run_id": int(path.parent.name),
                "check": m["check"],
                "check_index": int(m["index"]),
                "frame_path": path.relative_to(data_dir).as_posix(),
            }
        )
    frames = pd.DataFrame(
        rows, columns=["run_id", "check", "check_index", "frame_path"]
    )
    frames = frames.sort_values(["run_id", "check_index"], ignore_index=True)
    attempt = frames.groupby(["run_id", "check"]).cumcount() + 1
    phases = {r: run_phases(data_dir, r) for r in frames["run_id"].unique()}
    for i, row in frames.iterrows():
        seen = phases[row["run_id"]][: row["check_index"] + 1]
        if seen and seen[-1] == row["check"]:  # count missing frames too
            attempt[i] = seen.count(row["check"])
    frames.insert(2, "attempt", attempt)
    return frames.astype(
        {"run_id": "int64", "attempt": "int32", "check_index": "int32"}
    )


def run_phases(data_dir: Path, run_id: int) -> list[str]:
    """visionChecks phases in order from the raw run JSON, [] if absent."""
    path = data_dir / RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        return []
    run = json.loads(path.read_text(encoding="utf-8"))
    return [c.get("phase", "") for c in run.get("visionChecks") or []]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def embed_frames(
    frames: pd.DataFrame,
    data_dir: Path,
    model_path: Path = MODEL_PATH,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """frames plus e0..e{D-1} (float32) and model_sha256. Unreadable
    frames are logged and dropped."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    kept: list[int] = []
    chunks: list[np.ndarray] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        if batch:
            x = np.concatenate(batch)
            chunks.append(sess.run(None, {input_name: x})[0])
            batch.clear()

    for i, rel in enumerate(frames["frame_path"]):
        try:
            batch.append(preprocess(load_frame(data_dir / rel)))
        except ValueError as exc:
            log.warning("skipping frame: %s", exc)
            continue
        kept.append(i)
        if len(batch) == batch_size:
            flush()
    flush()

    out = frames.iloc[kept].reset_index(drop=True)
    dim = sess.get_outputs()[0].shape[-1]
    emb = (
        np.concatenate(chunks).astype(np.float32)
        if chunks
        else np.empty((0, dim), dtype=np.float32)
    )
    cols = pd.DataFrame(
        emb, columns=[f"{EMBED_PREFIX}{j}" for j in range(emb.shape[1])]
    )
    out = pd.concat([out, cols], axis=1)
    out["model_sha256"] = file_sha256(model_path)
    return out


def build_frame_embeddings(
    data_dir: Path,
    model_path: Path = MODEL_PATH,
    batch_size: int = BATCH_SIZE,
) -> Path:
    """Embed every raw frame and write the parquet cache."""
    frames = scan_frames(data_dir)
    table = embed_frames(frames, data_dir, model_path, batch_size)
    out = data_dir / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(
        f"{out}: {len(table)} of {len(frames)} frames embedded, "
        f"{table.shape[1] - len(frames.columns) - 1} dims"
    )
    return out
