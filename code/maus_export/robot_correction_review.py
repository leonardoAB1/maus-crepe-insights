"""Draw best and worst review sheets for robot correction estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .rectification import private_output, read_image
from .robot_correction import normalize_matrix

PANEL_WIDTH = 260
PANEL_HEIGHT = 260
LABEL_HEIGHT = 54
GAP = 8
JPEG_QUALITY = 92


def corners(size: list[int]) -> np.ndarray:
    """Return source image corners as 2D points."""
    width, height = size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def estimate_scores(report: dict) -> list[tuple[float, dict]]:
    """Score estimates by corner displacement from the mean transform."""
    matrix = normalize_matrix(np.array(report["matrix_mean"]))
    source_corners = corners(report["source_size"])
    mean_points = cv2.perspectiveTransform(source_corners[None], matrix)[0]
    scored = []
    for estimate in report["estimates"]:
        estimate_matrix = normalize_matrix(np.array(estimate["matrix"]))
        points = cv2.perspectiveTransform(
            source_corners[None], estimate_matrix
        )[0]
        displacement = np.linalg.norm(points - mean_points, axis=1)
        scored.append((float(displacement.max()), estimate))
    return sorted(scored, key=lambda item: item[0])


def fit_panel(image: np.ndarray) -> np.ndarray:
    """Fit an image into a fixed-size white panel."""
    height, width = image.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 245, dtype=np.uint8)
    y0 = (PANEL_HEIGHT - new_height) // 2
    x0 = (PANEL_WIDTH - new_width) // 2
    panel[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return panel


def labeled(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Add a small text label above a panel."""
    label = np.full((LABEL_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    for index, line in enumerate(lines[:3]):
        cv2.putText(
            label,
            line[:38],
            (6, 16 + index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([label, fit_panel(image)])


def diff_panel(corrected: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Return a heatmap of absolute pixel differences."""
    difference = cv2.absdiff(corrected, overlay)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mapped = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return mapped


def review_row(report: dict, estimate: dict, score: float) -> np.ndarray:
    """Draw raw, corrected, overlay and difference panels for one estimate."""
    raw = read_image(Path(estimate["frame"]))
    overlay = read_image(Path(estimate["overlay"]))
    matrix = normalize_matrix(np.array(report["matrix_mean"]))
    corrected = cv2.warpPerspective(raw, matrix, tuple(report["target_size"]))
    validation = estimate["validation"]
    title = f"{estimate['run_id']} {estimate['phase']} score {score:.1f}px"
    panels = [
        labeled(raw, ["raw", title, Path(estimate["frame"]).name]),
        labeled(
            corrected,
            [
                "mean corrected",
                f"median {validation['median_error_px']:.2f}px",
                f"p95 {validation['p95_error_px']:.1f}px",
            ],
        ),
        labeled(
            overlay,
            [
                "robot overlay target",
                f"inlier {validation['held_out_inlier_share']:.2f}",
                f"coverage {validation['held_out_coverage']:.2f}",
            ],
        ),
        labeled(
            diff_panel(corrected, overlay),
            ["abs diff heatmap", "corrected vs overlay", "bright = different"],
        ),
    ]
    spacer = np.full((panels[0].shape[0], GAP, 3), 255, dtype=np.uint8)
    row = panels[0]
    for panel in panels[1:]:
        row = np.hstack([row, spacer, panel])
    return row


def stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    """Stack rows with a small white separator."""
    sheet = rows[0]
    separator = np.full((GAP, sheet.shape[1], 3), 255, dtype=np.uint8)
    for row in rows[1:]:
        sheet = np.vstack([sheet, separator, row])
    return sheet


def write_sheet(
    report: dict, items: list[tuple[float, dict]], output: Path
) -> None:
    """Write one contact sheet."""
    rows = [review_row(report, estimate, score) for score, estimate in items]
    sheet = stack_rows(rows)
    cv2.imwrite(
        str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--correction", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.out.resolve()
    best = private_output(output_dir / "best.jpg", data_dir)
    worst = private_output(output_dir / "worst.jpg", data_dir)
    index = private_output(output_dir / "index.json", data_dir)
    report = json.loads(args.correction.read_text())
    scored = estimate_scores(report)
    count = min(args.count, len(scored))
    write_sheet(report, scored[:count], best)
    write_sheet(report, list(reversed(scored[-count:])), worst)
    payload = {
        "correction": str(args.correction),
        "best": str(best),
        "worst": str(worst),
        "count": count,
        "score": "max corner displacement from mean transform in pixels",
    }
    index.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
