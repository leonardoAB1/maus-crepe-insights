"""maus-export: fetch | mcap | table | all"""

from __future__ import annotations

import argparse
from pathlib import Path

import anyio

from .fetch import DEFAULT_FROM, DEFAULT_TO, fetch

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="maus-export",
        description="Export Maus crêpe runs to JSON, MCAP and tables",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="output folder (default: <repo>/data)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def fetch_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--from", dest="date_from", default=DEFAULT_FROM)
        sp.add_argument("--to", dest="date_to", default=DEFAULT_TO)
        sp.add_argument("--limit", type=int, help="only the newest N runs")
        sp.add_argument(
            "--hide-tests", action="store_true", help="skip test runs"
        )
        sp.add_argument(
            "--media",
            action="store_true",
            help="also download vision-check images",
        )
        sp.add_argument(
            "--video",
            action="store_true",
            help="also download run videos (large)",
        )
        sp.add_argument(
            "--refresh",
            action="store_true",
            help="re-download runs already on disk",
        )

    fetch_args(
        sub.add_parser(
            "fetch", help="download raw run JSON (+ media) from the dashboard"
        )
    )
    m = sub.add_parser(
        "mcap", help="convert raw runs to one ROS 2 MCAP per run"
    )
    m.add_argument(
        "--video",
        action="store_true",
        help="embed video.mp4 as an MCAP attachment",
    )
    m.add_argument("--ids", type=int, nargs="*", help="only these run ids")
    sub.add_parser("table", help="build runs.csv / runs.parquet")
    sub.add_parser("training", help="export candidate run-start dataset")
    sub.add_parser(
        "vision", help="score check overlays -> vision.csv / vision.parquet"
    )
    ex = sub.add_parser(
        "example", help="draw the annotated vision example figure"
    )
    ex.add_argument("--spread", type=int, help="run id for the spread panel")
    ex.add_argument("--fold", type=int, help="run id for the fold-2 panel")
    ex.add_argument(
        "--out",
        type=Path,
        default=Path("../assets/vision-scores-example.jpg"),
        help="output image (.jpg or .png)",
    )
    an = sub.add_parser(
        "analyze", help="which settings matter -> data/analysis/ (ml env)"
    )
    an.add_argument(
        "--boot", type=int, default=None, help="bootstrap draws (default 500)"
    )
    dd = sub.add_parser(
        "doe-design", help="DSD run sheet for Maus -> data/doe/"
    )
    dd.add_argument("--robot", required=True, help="robot that runs it")
    dd.add_argument("--topping", default="plain")
    dd.add_argument("--seed", type=int, default=0, help="run order seed")
    dd.add_argument("--fake", type=int, default=2, help="unused design columns")
    dd.add_argument(
        "--centre", type=int, default=3, help="centre runs in total"
    )
    da = sub.add_parser(
        "doe-analyze", help="effects from a filled run sheet (ml env)"
    )
    da.add_argument("sheet", type=Path, help="run sheet with run ids")
    df = sub.add_parser(
        "doe-figures", help="redraw the doe figures in assets/ (ml env)"
    )
    df.add_argument("--assets", type=Path, default=REPO_ROOT / "assets")
    a = sub.add_parser("all", help="fetch + mcap + table + vision")
    fetch_args(a)

    args = p.parse_args()
    data_dir: Path = args.data_dir

    if args.cmd in ("fetch", "all"):
        anyio.run(
            lambda: fetch(
                data_dir,
                args.date_from,
                args.date_to,
                args.limit,
                args.hide_tests,
                args.media,
                args.video,
                args.refresh,
            )
        )
    if args.cmd in ("mcap", "all"):
        from .to_mcap import convert_all

        convert_all(
            data_dir, include_video=args.video, only=getattr(args, "ids", None)
        )
    if args.cmd in ("table", "all"):
        from .to_table import build_table

        build_table(data_dir)
    if args.cmd == "training":
        from .training_dataset import (
            TrainingInputError,
            build_training_dataset,
        )

        failed = "Training export failed, previous output left unchanged"
        try:
            build_training_dataset(data_dir)
        except TrainingInputError as exc:
            # Names the file and a fixed reason only, so safe to print
            p.exit(1, f"{failed}: {exc}\n")
        except Exception as exc:
            # Other errors may quote private values, so print only the type
            p.exit(
                1,
                f"{failed} ({type(exc).__name__}). Check raw input locally.\n",
            )
    if args.cmd in ("vision", "all"):
        from .vision_metrics import build_vision_table

        build_vision_table(data_dir)
    if args.cmd == "analyze":
        from .analysis import N_BOOT, run_analysis

        run_analysis(data_dir, n_boot=args.boot or N_BOOT)
    if args.cmd == "doe-design":
        from .doe import build_run_sheet

        build_run_sheet(
            data_dir,
            args.robot,
            args.topping,
            args.seed,
            args.fake,
            args.centre,
        )
    if args.cmd == "doe-analyze":
        from .doe_analyze import analyze

        analyze(data_dir, args.sheet)
    if args.cmd == "doe-figures":
        from .doe_figures import example_figures

        for path in example_figures(data_dir, args.assets):
            print(path)
    if args.cmd == "example":
        from .figures import (
            DEFAULT_FOLD_RUN,
            DEFAULT_SPREAD_RUN,
            vision_example,
        )

        vision_example(
            data_dir,
            args.out,
            args.spread or DEFAULT_SPREAD_RUN,
            args.fold or DEFAULT_FOLD_RUN,
        )


if __name__ == "__main__":
    main()
