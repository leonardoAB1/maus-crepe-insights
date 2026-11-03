"""Build the pitch deck: assets/pitch/index.html from template.html.

Inlines reveal.js's stylesheet and the images from assets/ as data URIs.
reveal.js itself, the QR code library and the fonts load from public CDNs,
so the page needs internet. The PDF next to it works offline.

Usage (from the repo root): python assets/pitch/source/build.py
"""

import base64
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PITCH = HERE.parent
ASSETS = PITCH.parent
REVEAL_CSS = HERE / "vendor" / "reveal.min.css"
REVEAL_CSS_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.1.0/reveal.min.css"
)
MIME = {".jpg": "image/jpeg", ".png": "image/png"}
EM_DASH = "—"


def reveal_css() -> str:
    """Return reveal.js's stylesheet, downloading it once if needed."""
    if not REVEAL_CSS.is_file():
        REVEAL_CSS.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REVEAL_CSS_URL) as response:
            REVEAL_CSS.write_bytes(response.read())
    return REVEAL_CSS.read_text(encoding="utf-8")


def data_uri(match: re.Match) -> str:
    path = ASSETS / match.group(1)
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{MIME[path.suffix]};base64,{encoded}"


def main() -> None:
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*@@REVEAL_CSS@@*/", reveal_css())
    html = re.sub(r"@@IMG:([\w.-]+)@@", data_uri, html)
    html = html.replace("<!--@@VIDEO-->", "").replace("<!--@@/VIDEO-->", "")
    assert "@@" not in html and EM_DASH not in html
    page = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1">\n' + html + "\n</html>\n"
    )
    with open(PITCH / "index.html", "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print(PITCH / "index.html", len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
