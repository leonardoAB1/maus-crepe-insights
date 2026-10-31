"""maus-classify: frames | coverage | fold-class"""

from __future__ import annotations

import argparse
from pathlib import Path

from maus_export.cli import REPO_ROOT

from .frames import FramesError, build_frames


def main() -> None:
    p = argparse.ArgumentParser(
        prog="maus-classify",
        description="Datasets and harness for vision check classifiers",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="export folder with raw/runs (default: <repo>/data)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "frames", help="one row per check frame -> training/frames.parquet"
    )
    sub.add_parser(
        "coverage", help="Markdown tables of data coverage (aggregates)"
    )
    fc = sub.add_parser(
        "fold-class", help="fold state head -> models/fold-class (ml env)"
    )
    fc.add_argument("--out", type=Path, default=None, help="model folder")
    fx = sub.add_parser(
        "fold-class-example", help="example grid -> assets/ (ml env)"
    )
    fx.add_argument("--out", type=Path, default=None, help="image path")
    args = p.parse_args()

    if args.cmd == "coverage":
        from .coverage import CoverageError, main_data_dir, report

        data_dir = args.data_dir
        if not (data_dir / "raw" / "runs").is_dir():
            # A worktree has no export of its own; use the main checkout's
            data_dir = main_data_dir() or data_dir
        try:
            print(report(data_dir), end="")
        except CoverageError as exc:
            p.exit(1, f"Coverage report failed: {exc}\n")

    if args.cmd == "fold-class-example":
        from .fold_class import EXAMPLE_ASSET, example_figure

        path = example_figure(
            args.data_dir,
            REPO_ROOT / "code" / "models" / "fold-class",
            args.out or REPO_ROOT / EXAMPLE_ASSET,
        )
        print(f"example figure -> {path}")

    if args.cmd == "fold-class":
        from .fold_class import FoldClassError, train_fold_class

        out = args.out or REPO_ROOT / "code" / "models" / "fold-class"
        try:
            card = train_fold_class(args.data_dir, out)
        except FoldClassError as exc:
            p.exit(1, f"Fold class training failed: {exc}\n")
        conf = card["metrics"]["test"]["confidence"]
        print(
            f"{out}: holdout accuracy {conf['accuracy']:.3f}, "
            f"mean confidence {conf['mean_confidence']:.3f}"
        )

    if args.cmd == "frames":
        failed = "Frame export failed, previous output left unchanged"
        try:
            build_frames(args.data_dir)
        except FramesError as exc:
            # Fixed messages only, so safe to print
            p.exit(1, f"{failed}: {exc}\n")
        except Exception as exc:
            # Other errors may quote private values, so print only the type
            p.exit(
                1,
                f"{failed} ({type(exc).__name__}). Check raw input locally.\n",
            )


if __name__ == "__main__":
    main()
