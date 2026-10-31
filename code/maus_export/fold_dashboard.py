"""Local HTML review dashboard for fold-segmentation videos."""

from __future__ import annotations

import html
import json
from pathlib import Path


def build_dashboard(data_dir: Path) -> Path:
    """Build a local static dashboard for generated live videos."""
    root = data_dir / "fold_segmentation"
    live = root / "live"
    rows = []
    if live.exists():
        for run_dir in sorted(path for path in live.iterdir() if path.is_dir()):
            summary_path = run_dir / "summary.json"
            summary = {}
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": run_dir.name,
                    "video": (run_dir / "annotated.mp4").relative_to(root),
                    "timeline": (run_dir / "timeline.csv").relative_to(root),
                    "plot": (run_dir / "timeline.png").relative_to(root),
                    "median_ms": summary.get("median_ms_per_frame"),
                }
            )
    content = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Fold segmentation review</title>",
        "<style>",
        "body{font-family:sans-serif;margin:2rem;}",
        "article{border-top:1px solid #ccc;padding:1rem 0;}",
        "video{max-width:760px;width:100%;display:block;}",
        "img{max-width:760px;width:100%;cursor:pointer;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Fold segmentation review</h1>",
        "<p>Generated locally from private artifacts under data/.</p>",
        "<h2>Overview</h2>",
        "<ul>",
    ]
    for row in rows:
        median = row["median_ms"]
        median_text = "n/a" if median is None else f"{float(median):.1f} ms"
        run_id = html.escape(row["run_id"])
        content.append(
            f'<li><a href="#{run_id}">run {run_id}</a>: {median_text}</li>'
        )
    content.extend(["</ul>", "<h2>Runs</h2>"])
    for row in rows:
        run_id = html.escape(row["run_id"])
        video = html.escape(row["video"].as_posix())
        plot = html.escape(row["plot"].as_posix())
        timeline = html.escape(row["timeline"].as_posix())
        content.extend(
            [
                f'<article id="{run_id}">',
                f"<h3>Run {run_id}</h3>",
                f'<video controls src="{video}"></video>',
                (f'<img src="{plot}" data-video="{run_id}" alt="timeline">'),
                f'<p><a href="{timeline}">timeline csv</a></p>',
                "</article>",
            ]
        )
    content.extend(
        [
            "<script>",
            "document.querySelectorAll('img[data-video]').forEach(img=>{",
            "img.addEventListener('click', event=>{",
            "const video=img.parentElement.querySelector('video');",
            "const rect=img.getBoundingClientRect();",
            "const share=(event.clientX-rect.left)/rect.width;",
            "if(video.duration){video.currentTime=share*video.duration;video.play();}",
            "});});",
            "</script>",
            "</body></html>",
        ]
    )
    path = root / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")
    return path
