"""Committed notebooks must have no outputs: the repo is public and outputs
would carry run-level data. Clear them with
`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`."""

import json
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p for p in CODE_DIR.rglob("*.ipynb") if ".pixi" not in p.parts
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"cell {i} has outputs"
        assert cell.get("execution_count") is None, f"cell {i} was run"
