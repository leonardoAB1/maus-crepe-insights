"""Estimate one mean image correction for a robot from local media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .rectification import private_output
from .robot_correction import (
    DEFAULT_PHASES,
    aggregate_transforms,
    collect_candidates,
    estimate_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--robot",
        required=True,
        help="machine name such as maus-pixi, or numeric machine id",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help=(
            "vision phase to use; repeatable, defaults to spread/folds/push_off"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum candidate pairs to try after sorting by run and phase",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out = private_output(args.out, data_dir)
    cv2.setNumThreads(1)
    phases = tuple(args.phases) if args.phases else DEFAULT_PHASES
    candidates = collect_candidates(data_dir, args.robot, phases)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    accepted = []
    rejected = []
    for candidate in candidates:
        try:
            accepted.append(estimate_candidate(candidate))
        except (ValueError, cv2.error) as error:
            rejected.append(
                {
                    "run_id": candidate.run_id,
                    "phase": candidate.phase,
                    "frame": str(candidate.frame),
                    "overlay": str(candidate.overlay),
                    "reason": str(error),
                }
            )
    report = aggregate_transforms(accepted, args.robot)
    report["candidate_pairs"] = len(candidates)
    report["rejected_estimates"] = len(rejected)
    report["rejections"] = rejected
    out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "robot": args.robot,
                "candidate_pairs": len(candidates),
                "accepted_estimates": len(accepted),
                "rejected_estimates": len(rejected),
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
