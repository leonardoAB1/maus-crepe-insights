"""One-command private Label Studio workspaces for assigned labeling sets."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn
from urllib.error import URLError
from urllib.request import urlopen

if __package__:
    from .assignments import (
        SET_NUMBERS,
        bundle_workspace,
        private_path,
        receive_bundle,
        split_pilot,
        validate_workspace,
    )
    from .workflow import HUMAN_TAXONOMY_VERSION, MASK_TAXONOMY_VERSION, runtime
else:
    from assignments import (
        SET_NUMBERS,
        bundle_workspace,
        private_path,
        receive_bundle,
        split_pilot,
        validate_workspace,
    )
    from workflow import HUMAN_TAXONOMY_VERSION, MASK_TAXONOMY_VERSION, runtime

BASE_PORT = 8081
READY_TIMEOUT = 120
SIGNUP_TIMEOUT = 900
POLL_SECONDS = 0.5
STOP_TIMEOUT = 20


def _interrupt(*_args: object) -> NoReturn:
    raise KeyboardInterrupt


def _owner(app: Path) -> str | None:
    database = app / "label_studio.sqlite3"
    if not database.exists():
        return None
    try:
        with sqlite3.connect(
            "file:" + str(database) + "?mode=ro", uri=True
        ) as db:
            rows = db.execute(
                "select email from htx_user where is_active=1 "
                "and active_organization_id is not null"
            ).fetchall()
    except sqlite3.OperationalError as error:
        if "no such table" in str(error):
            return None
        raise
    if len(rows) > 1:
        raise ValueError("Use a single-user local app for this assignment")
    return rows[0][0] if rows else None


def _ready(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Label Studio stopped; inspect private app/server.log"
            )
        try:
            with urlopen(url + "/user/login/", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(POLL_SECONDS)
    raise TimeoutError(
        "Label Studio did not become ready; inspect app/server.log"
    )


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_TIMEOUT)


def setup_assignment(pilot: Path, app: Path, port: int, email: str) -> dict:
    """Idempotently install separate phase/mask projects for the local owner."""
    command, env = runtime(pilot, app, port)
    workflow = Path(__file__).with_name("workflow.py").resolve()
    code = (
        "import json,runpy\nfrom pathlib import Path\n"
        f"w=runpy.run_path({str(workflow)!r})\n"
        f"pilot=Path({str(pilot)!r})\n"
        f"phase=w['setup_project'](pilot,{email!r})\n"
        f"masks=w['setup_project'](pilot,{email!r},kind='masks')\n"
        "print('ASSIGNMENT_SETUP='+json.dumps({'phase':phase,'masks':masks}))\n"
    )
    with (app / ".assignment-setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(
            command[:3] + ["shell"],
            input=f"exec({code!r})\n",
            capture_output=True,
            text=True,
            env=env,
            cwd=app,
            timeout=READY_TIMEOUT,
        )
        (app / "assignment-setup.log").write_text(result.stdout + result.stderr)
    summaries = [
        line.split("ASSIGNMENT_SETUP=", 1)[1]
        for line in result.stdout.splitlines()
        if "ASSIGNMENT_SETUP=" in line
    ]
    if result.returncode or not summaries:
        raise RuntimeError(
            "Project setup failed; inspect app/assignment-setup.log"
        )
    return json.loads(summaries[-1])


def start_workspace(
    workspace: Path, number: int, port: int | None = None
) -> None:
    """Run this set and automatically install projects after first signup."""
    os.umask(0o077)
    workspace = private_path(workspace)
    plan = validate_workspace(workspace, number)
    pilot, app = workspace / "pilot", workspace / "app"
    port = port if port is not None else BASE_PORT + number
    state_path = workspace / "workspace.json"
    identity = {
        "pilot_sha256": plan["digest"],
        "assignment": plan["assignment"],
        "app_path": str(app),
        "phase_taxonomy": HUMAN_TAXONOMY_VERSION,
        "mask_taxonomy": MASK_TAXONOMY_VERSION,
    }
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Workspace changed or moved; do not migrate its DB blindly"
            )
    command, env = runtime(pilot, app, port)
    url = f"http://localhost:{port}"
    with (workspace / ".launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", port))
        owner = _owner(app)
        first_signup = owner is None
        if first_signup:
            env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "false"
        with (app / "server.log").open("a") as log:
            process = subprocess.Popen(
                command, env=env, cwd=app, stdout=log, stderr=log
            )
            try:
                _ready(process, url)
                print(f"Set {number}: {url}", flush=True)
                if first_signup:
                    print(
                        "Create your local account there. "
                        "Project import follows automatically.",
                        flush=True,
                    )
                    deadline = time.monotonic() + SIGNUP_TIMEOUT
                    while owner is None and time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(
                                "App stopped while waiting for signup"
                            )
                        time.sleep(POLL_SECONDS)
                        owner = _owner(app)
                    if owner is None:
                        raise TimeoutError(
                            "No local signup detected; run this command again"
                        )
                projects = setup_assignment(pilot, app, port, owner)
                if first_signup:
                    _stop(process)
                    env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "true"
                    process = subprocess.Popen(
                        command, env=env, cwd=app, stdout=log, stderr=log
                    )
                    _ready(process, url)
                state = {**identity, "projects": projects, "port": port}
                temporary = state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state, indent=2))
                temporary.replace(state_path)
                print(
                    json.dumps(
                        {
                            "set_number": number,
                            "tasks_each": len(plan["tasks"]),
                            "phase_url": url
                            + f"/projects/{projects['phase']['project_id']}",
                            "mask_url": url
                            + f"/projects/{projects['masks']['project_id']}",
                        }
                    ),
                    flush=True,
                )
                print(
                    "Keep this command running. "
                    "Ctrl-C stops the app, not your saved work.",
                    flush=True,
                )
                process.wait()
            finally:
                _stop(process)


def main() -> int:
    """Split, hand off, or start one assignment without dashboard access."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    split = actions.add_parser("split")
    split.add_argument("--pilot", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    for action in ("bundle", "receive", "start"):
        child = actions.add_parser(action)
        child.add_argument(
            "--set", dest="number", type=int, choices=SET_NUMBERS, required=True
        )
        if action != "receive":
            child.add_argument("--workspace", type=Path, required=True)
        if action in {"bundle", "receive"}:
            child.add_argument("--output", type=Path, required=True)
        if action == "receive":
            child.add_argument("--bundle", type=Path, required=True)
            child.add_argument("--sha256", required=True)
        if action == "start":
            child.add_argument("--port", type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        if args.action == "split":
            result = split_pilot(args.pilot, args.output)
        elif args.action == "bundle":
            result = bundle_workspace(args.workspace, args.number, args.output)
        elif args.action == "receive":
            result = receive_bundle(
                args.bundle, args.sha256, args.number, args.output
            )
        else:
            start_workspace(args.workspace, args.number, args.port)
            return 0
    except KeyboardInterrupt:
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        TimeoutError,
    ) as error:
        print(f"Labeling assignment failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
