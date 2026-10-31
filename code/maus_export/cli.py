"""maus-export: fetch | mcap | table | all"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anyio

from .fetch import DEFAULT_FROM, DEFAULT_TO, fetch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Coordinate spaces and the default score threshold live in label_deviation,
# which needs numpy, OpenCV and pandas. They are repeated here so every CLI
# command does not pay that import cost; label_deviation validates them again.
LABEL_SPACES = ("rectified_upstream", "native")
DEFAULT_SCORE_THRESHOLD = 0.35


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
    sub.add_parser("vision-targets", help="join run-start and vision targets")
    sub.add_parser("audit", help="audit training and vision coverage")
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
    em = sub.add_parser(
        "embed", help="cache frame embeddings -> data/training/ (ml env)"
    )
    em.add_argument("--model", type=Path, default=None, help="model.onnx")
    em.add_argument("--batch-size", type=int, default=None)
    ls = sub.add_parser(
        "label-states", help="print the unified crepe-state vocabulary"
    )
    ls.add_argument(
        "--out", type=Path, default=None, help="write the JSON here as well"
    )
    ld = sub.add_parser(
        "label-deviation",
        help="deviation between human masks and model predictions",
    )
    ld.add_argument(
        "--labels", type=Path, required=True, help="labeling mask export"
    )
    ld.add_argument(
        "--predictions", type=Path, required=True, help="prediction index"
    )
    ld.add_argument("--out", type=Path, required=True, help="report folder")
    ld.add_argument("--space", choices=LABEL_SPACES, default=LABEL_SPACES[0])
    ld.add_argument(
        "--prediction-space", choices=LABEL_SPACES, default=LABEL_SPACES[0]
    )
    ld.add_argument("--pilot", type=Path, default=None)
    ld.add_argument("--transforms", type=Path, default=None)
    ld.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    ld.add_argument("--worst", type=int, default=25)
    fd = sub.add_parser(
        "finetune-dataset",
        help="weighted, run-grouped dataset from a deviation report",
    )
    fd.add_argument("--deviation", type=Path, required=True)
    fd.add_argument("--labels", type=Path, required=True)
    fd.add_argument("--predictions", type=Path, default=None)
    fd.add_argument("--out", type=Path, required=True)
    ov = sub.add_parser(
        "overlay", help="draw the pipeline layers onto a video or a stream"
    )
    ov.add_argument(
        "--input",
        required=True,
        help="video file, or device:N for a camera or stream index",
    )
    ov.add_argument("--output", type=Path, required=True, help=".mp4 output")
    ov.add_argument("--records", type=Path, default=None)
    ov.add_argument("--masks-dir", type=Path, default=None)
    ov.add_argument("--run-id", type=int, default=None)
    ov.add_argument("--fps", type=float, default=15.0)
    ov.add_argument("--start", type=int, default=0)
    ov.add_argument("--limit", type=int, default=None)
    vw = sub.add_parser(
        "view", help="watch a run with the pipeline layers beside it"
    )
    vw.add_argument("--video", type=Path, required=True)
    vw.add_argument("--records", type=Path, default=None)
    vw.add_argument("--masks-dir", type=Path, default=None)
    vw.add_argument("--producer", default="pixel_masks")
    vw.add_argument("--run-id", type=int, default=None)
    vw.add_argument("--host", default="127.0.0.1")
    vw.add_argument("--port", type=int, default=8087)
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
    if args.cmd == "vision-targets":
        from .vision_targets import VisionTargetsError, build_vision_targets

        try:
            build_vision_targets(data_dir)
        except VisionTargetsError as exc:
            p.exit(
                1,
                "Vision-target export failed; previous output is stale: "
                f"{exc}.\n",
            )
        except Exception as exc:
            p.exit(
                1,
                "Vision-target export failed; previous output is stale "
                f"({type(exc).__name__}). Check inputs locally.\n",
            )
    if args.cmd == "audit":
        from .training_audit import build_training_audit

        try:
            build_training_audit(data_dir)
        except Exception as exc:
            p.exit(
                1,
                "Training audit failed; previous output is stale "
                f"({type(exc).__name__}). Check inputs locally.\n",
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
    if args.cmd == "embed":
        from .frame_embeddings import (
            BATCH_SIZE,
            MODEL_PATH,
            build_frame_embeddings,
        )

        build_frame_embeddings(
            data_dir,
            args.model or MODEL_PATH,
            args.batch_size or BATCH_SIZE,
        )
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
    if args.cmd == "label-states":
        from .crepe_states import state_table

        table = state_table()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(table, indent=2) + "\n")
        print(json.dumps(table, indent=2))
    if args.cmd == "label-deviation":
        from .label_deviation import run_comparison

        out: Path = args.out
        report = run_comparison(
            labels_directory=args.labels,
            prediction_index=args.predictions,
            output=out,
            space=args.space,
            prediction_space=args.prediction_space,
            pilot=args.pilot,
            transforms=args.transforms,
            score_threshold=args.score_threshold,
            worst_frames=args.worst,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    if args.cmd == "finetune-dataset":
        from .finetune_dataset import build_dataset, usage_summary

        dataset_out: Path = args.out
        manifest = build_dataset(
            deviation_directory=args.deviation,
            labels_directory=args.labels,
            predictions_directory=args.predictions,
            output=dataset_out,
        )
        print(usage_summary(manifest))
    if args.cmd == "overlay":
        from .overlay import load_records, render_video

        overlay_out: Path = args.output
        records = None
        if args.records is not None:
            records = load_records(args.records)
        result = render_video(
            source=args.input,
            output=overlay_out,
            records=records,
            masks_dir=args.masks_dir,
            run_id=args.run_id,
            fps=args.fps,
            start=args.start,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
    if args.cmd == "view":
        from .viewer import ViewerApp, load_record_file, serve

        records = (
            None if args.records is None else load_record_file(args.records)
        )
        app = ViewerApp(
            video=args.video,
            records=records,
            masks_dir=args.masks_dir,
            producer=args.producer,
            run_id=args.run_id,
        )
        serve(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
