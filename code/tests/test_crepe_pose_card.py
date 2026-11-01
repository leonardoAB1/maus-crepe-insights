"""The crepe-pose card must describe the model file next to it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

FOLDER = Path(__file__).resolve().parents[1] / "models" / "crepe-pose"
MODEL = FOLDER / "model.onnx"
CARD = FOLDER / "model.json"


@pytest.fixture
def card() -> dict:
    if not (MODEL.is_file() and CARD.is_file()):
        pytest.skip("the crepe-pose model is not in this checkout")
    return json.loads(CARD.read_text())


def test_card_names_the_model_inputs_and_outputs(card: dict) -> None:
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(MODEL))
    inputs = session.get_inputs()
    assert [i.name for i in inputs] == list(card["inputs"])
    assert inputs[0].shape[1:] == card["inputs"]["x"]["shape"][1:]
    assert [o.name for o in session.get_outputs()] == list(card["outputs"])


def test_card_matches_the_model_file(card: dict) -> None:
    content = MODEL.read_bytes()
    assert card["size"]["bytes"] == len(content)
    assert card["size"]["sha256"] == hashlib.sha256(content).hexdigest()


def test_card_does_not_claim_more_than_was_measured(card: dict) -> None:
    assert card["deployable"] is False
    assert "not used" in card["status"]
    assert card["generalization"] == "not established"
    assert "not measured" in card["latency"]["pi5_estimate_method"].lower()
    assert any("fold angle" in item for item in card["out_of_scope"])
    assert card["missing"]


def test_card_holds_no_run_identifiers(card: dict) -> None:
    assert "run_ids" not in card["data"]["hand_masks"]
    assert isinstance(card["metrics"]["hand_masks_held_out"]["per_run"], list)
