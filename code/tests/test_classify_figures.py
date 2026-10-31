from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_classify.figures import MATCH, MISS, Example, _caption, example_grid

SHAPE = (60, 80, 3)
MAX_BYTES = 400_000


def example(truth: str, predicted: str, note: str = "") -> Example:
    img = np.full(SHAPE, 128, dtype=np.uint8)
    return Example(img, truth, predicted, 0.875, "robot-a", note)


def test_caption_has_label_answer_confidence_robot() -> None:
    text = _caption(example("half (fold2 did nothing)", "quarter", "FAIL"))
    assert text.splitlines() == [
        "robot: half (fold2 did nothing)",
        "model: quarter (0.88)",
        "robot-a",
        "FAIL",
    ]


def test_example_grid_writes_jpeg(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "grid.jpg"
    tiles = [example("flat", "flat"), example("half", "quarter")] * 2
    cropped = []

    def crop(img: np.ndarray) -> np.ndarray:
        cropped.append(img.shape)
        return img[:, : SHAPE[1] // 2]

    assert example_grid(tiles, out, "title", "subtitle", crop=crop) == out
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > SHAPE[0]
    assert out.stat().st_size < MAX_BYTES
    assert len(cropped) == len(tiles)
    assert MATCH != MISS


def test_example_grid_needs_examples(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        example_grid([], tmp_path / "x.jpg", "t")


def test_agree_overrides_text_comparison(tmp_path: Path, monkeypatch) -> None:
    from maus_classify import figures

    colours = []
    real = figures.plt.Axes.set_title

    def spy(self, *args, **kwargs):
        colours.append(kwargs.get("color"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(figures.plt.Axes, "set_title", spy)
    tiles = [
        example("fail (missed)", "pass"),  # text fallback: red
        Example(
            np.zeros(SHAPE, np.uint8),
            "fail (missed)",
            "fail",
            0.9,
            "robot-a",
            agree=True,
        ),  # explicit: green
        Example(
            np.zeros(SHAPE, np.uint8),
            "pass",
            "pass",
            0.9,
            "robot-a",
            agree=False,
        ),  # explicit: red
    ]
    example_grid(tiles, tmp_path / "g.jpg", "t")
    assert colours == [MISS, MATCH, MISS]
