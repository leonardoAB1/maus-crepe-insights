"""maus-classify: frames | coverage | fold-class | robustness"""

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
    sub.add_parser(
        "model-examples",
        help="check-verdict and frame-embedder images -> assets/ (ml env)",
    )
    sub.add_parser(
        "robustness",
        help="heads under light, steam and rotation -> data/robustness (ml)",
    )
    sub.add_parser(
        "robustness-example", help="robustness failure image -> assets/ (ml)"
    )
    sub.add_parser(
        "fold-crosscheck",
        help="fold-class head vs fold segmentation -> data/robustness (ml)",
    )
    args = p.parse_args()

    if args.cmd == "robustness":
        from .robustness import build_robustness

        report = build_robustness(args.data_dir, REPO_ROOT / "code" / "models")
        for name, entry in report["perturbations"].items():
            fold, verdict = entry["fold"]["all"], entry["verdict"]["all"]
            print(
                f"{name:16s} fold bacc {fold['balanced_accuracy']:.3f} "
                f"flip {fold['flip_rate']:.3f} | verdict bacc "
                f"{verdict['balanced_accuracy']:.3f} "
                f"flip {verdict['flip_rate']:.3f}"
            )

    if args.cmd == "robustness-example":
        from .robustness import EXAMPLE_ASSET, example_figure

        path = example_figure(args.data_dir, REPO_ROOT / EXAMPLE_ASSET)
        print(f"example figure -> {path}")

    if args.cmd == "fold-crosscheck":
        from .fold_crosscheck import build_crosscheck

        report = build_crosscheck(
            args.data_dir, REPO_ROOT / "code" / "models" / "fold-class"
        )
        valid = report["their_validity_on_our_frames"]["agreement"]
        held = report["agreement"]["holdout"]["passed"]["agreement"]
        print(
            f"segmentation vs passed checks {valid:.3f}, "
            f"head vs segmentation (holdout) {held:.3f}"
        )

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

    if args.cmd == "model-examples":
        from .examples import (
            CHECK_VERDICT_ASSET,
            EMBEDDER_ASSET,
            check_verdict_example,
            embedder_example,
        )

        models = REPO_ROOT / "code" / "models"
        for path in (
            check_verdict_example(
                args.data_dir,
                models / "check-verdict",
                REPO_ROOT / CHECK_VERDICT_ASSET,
            ),
            embedder_example(args.data_dir, REPO_ROOT / EMBEDDER_ASSET),
        ):
            print(f"example figure -> {path}")

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
