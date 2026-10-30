"""Download crêpe runs (optionally check images and videos) as raw JSON."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import httpx2

from .client import MausClient, connect

# The account's visible window on the dashboard (nothing outside it).
DEFAULT_FROM = "2026-08-30"
DEFAULT_TO = "2026-12-30"
PAGE = 500


async def list_run_ids(
    client: MausClient, date_from: str, date_to: str, hide_tests: bool
) -> list[int]:
    ids: list[int] = []
    offset = 0
    while True:
        page = await client.call(
            "crepe_list",
            **{"from": date_from, "to": date_to},
            limit=PAGE,
            offset=offset,
            hideTests=hide_tests,
        )
        items = page.get("items", [])
        ids += [it["id"] for it in items]
        offset += len(items)
        if not items or offset >= page.get("total", 0):
            return ids


async def _download(http: httpx2.AsyncClient, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    r = await http.get(url)
    if r.status_code != 200:
        print(f"  ! {r.status_code} for {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


async def fetch(
    data_dir: Path,
    date_from: str = DEFAULT_FROM,
    date_to: str = DEFAULT_TO,
    limit: int | None = None,
    hide_tests: bool = False,
    media: bool = False,
    video: bool = False,
    refresh: bool = False,
    concurrency: int = 4,
) -> None:
    runs_dir = data_dir / "raw" / "runs"
    media_dir = data_dir / "raw" / "media"
    runs_dir.mkdir(parents=True, exist_ok=True)

    async with connect() as client:
        robots = await client.call("robot_list")
        (data_dir / "raw" / "robots.json").write_text(
            json.dumps(robots, indent=2), encoding="utf-8"
        )

        ids = await list_run_ids(client, date_from, date_to, hide_tests)
        if limit:
            ids = ids[:limit]
        print(f"{len(ids)} runs between {date_from} and {date_to}")

        token = client.bearer_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        sem = anyio.Semaphore(concurrency)
        done = 0

        async with httpx2.AsyncClient(
            headers=headers, timeout=120.0, follow_redirects=True
        ) as http:

            async def one(run_id: int) -> None:
                nonlocal done
                async with sem:
                    path = runs_dir / f"{run_id}.json"
                    if path.exists() and not refresh:
                        run = json.loads(path.read_text(encoding="utf-8"))
                    else:
                        run = await client.call("crepe_get", id=run_id)
                        path.write_text(
                            json.dumps(run, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if media:
                        for name, url in (
                            run.get("checkImageUrls") or {}
                        ).items():
                            await _download(
                                http, url, media_dir / str(run_id) / name
                            )
                    if video and run.get("videoUrl"):
                        await _download(
                            http,
                            run["videoUrl"],
                            media_dir / str(run_id) / "video.mp4",
                        )
                    done += 1
                    if done % 25 == 0 or done == len(ids):
                        print(f"  {done}/{len(ids)}")

            async with anyio.create_task_group() as tg:
                for run_id in ids:
                    tg.start_soon(one, run_id)
