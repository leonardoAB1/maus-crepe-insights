"""Render assets/pitch/index.html to crepeflow-pitch.pdf, one slide a page.

Each slide is screenshotted at 2x with headless Chrome or Edge, in the
light theme, and the pages are joined with Pillow. Chrome's own
print-to-PDF of reveal.js proved unreliable, so this takes pictures instead;
the PDF text is therefore not selectable.

Usage (from the repo root): python assets/pitch/source/make_pdf.py
Set CHROME to the browser executable if it is not found automatically.
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PITCH = Path(__file__).resolve().parent.parent
PAGE_W, PAGE_H = 1344, 756  # slide 1280x720 plus reveal's 5% margin
SCALE = 2
WAIT_MS = 6000
WORKERS = 5
BROWSERS = (
    "chrome",
    "google-chrome",
    "chromium",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser() -> str:
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    for name in BROWSERS:
        found = shutil.which(name) or (name if Path(name).is_file() else None)
        if found:
            return found
    raise SystemExit("Chrome or Edge not found; set CHROME")


def main() -> None:
    browser = find_browser()
    src = (PITCH / "index.html").read_text(encoding="utf-8")
    slides = src.count("<section")
    src = (
        src.replace('<html lang="en">', '<html lang="en" data-theme="light">')
        .replace("hash: false", "hash: true")
        .replace("controls: true", "controls: false")
        .replace("progress: true", "progress: false")
        .replace('transition: "fade"', 'transition: "none"')
        # Videos show their poster frame, without player controls.
        .replace(" controls preload", " preload")
        .replace("data-start=", "data-nostart=")
    )
    render = PITCH / "_render.html"
    render.write_text(src, encoding="utf-8")
    shots = Path(tempfile.mkdtemp())

    def shoot(n: int) -> Path:
        out = shots / f"{n:02d}.png"
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--window-size={PAGE_W},{PAGE_H}",
             f"--force-device-scale-factor={SCALE}",
             f"--timeout={WAIT_MS}", f"--screenshot={out}",
             render.as_uri() + f"#/{n}"],
            capture_output=True, check=False,
        )
        return out

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            paths = list(pool.map(shoot, range(slides)))
        images = [Image.open(p).convert("RGB") for p in paths]
        pdf = PITCH / "crepeflow-pitch.pdf"
        images[0].save(pdf, save_all=True, append_images=images[1:],
                       resolution=72 * SCALE)
        print(pdf, len(images), "pages")
    finally:
        render.unlink(missing_ok=True)
        shutil.rmtree(shots, ignore_errors=True)


if __name__ == "__main__":
    main()
