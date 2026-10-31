"""Apply a mean robot image correction to files or a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .rectification import private_output
from .robot_correction import (
    apply_correction_to_image,
    image_inputs,
    output_path_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--correction", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--write-valid-mask",
        action="store_true",
        help="also write *_valid.png masks for supported output pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.out.resolve()
    correction = json.loads(args.correction.read_text())
    input_root, inputs = image_inputs(args.input.resolve())
    written = 0
    skipped = []
    for source in inputs:
        destination = output_path_for(source, input_root, output_root)
        try:
            private_output(destination, data_dir)
            corrected, valid = apply_correction_to_image(source, correction)
            cv2.imwrite(str(destination), corrected)
            if args.write_valid_mask:
                mask = destination.with_name(destination.stem + "_valid.png")
                private_output(mask, data_dir)
                cv2.imwrite(str(mask), valid)
            written += 1
        except (ValueError, cv2.error) as error:
            skipped.append({"source": str(source), "reason": str(error)})
    report = {
        "correction": str(args.correction),
        "input": str(args.input),
        "out": str(output_root),
        "written_images": written,
        "skipped_images": len(skipped),
        "skips": skipped,
    }
    report_path = private_output(output_root / "apply_report.json", data_dir)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
