"""Tests for the boundary-zone fold-state classifier."""

import math

import numpy as np
import pytest

from maus_export.fold_classifier import (
    DEFAULT_BIAS,
    DEFAULT_MEAN,
    DEFAULT_SCALE,
    DEFAULT_WEIGHTS,
    FEATURES,
    ZONE_HIGH,
    ZONE_LOW,
    fit_logistic,
    half_probability,
    refine_fold_state,
)

HALF_FEATURES = {
    "relative": 0.5,
    "plate_fraction": 0.4,
    "rect_aspect": 2.0,
    "diameter_fraction": 0.6,
    "offset_fraction": 0.1,
    "hole_fraction": 0.0,
    "solidity": 0.95,
    "circularity": 0.7,
    "edge_count": 3,
}
QUARTER_FEATURES = {
    "relative": 0.2,
    "plate_fraction": 0.25,
    "rect_aspect": 1.2,
    "diameter_fraction": 0.5,
    "offset_fraction": 0.15,
    "hole_fraction": 0.0,
    "solidity": 0.93,
    "circularity": 0.65,
    "edge_count": 2,
}


def _features(**overrides):
    merged = {name: HALF_FEATURES[name] for name in FEATURES}
    merged.update(overrides)
    return merged


def test_fit_logistic_separates_two_blobs() -> None:
    rng = np.random.default_rng(7)
    low = rng.normal(0.0, 0.2, size=(60, 3))
    high = rng.normal(3.0, 0.2, size=(60, 3))
    x = np.vstack([low, high])
    y = np.array([0.0] * 60 + [1.0] * 60)
    weights, bias, mean, scale = fit_logistic(x, y, iterations=800)
    proba_low = half_probability(weights, bias, mean, scale, tuple(low[0]))
    proba_high = half_probability(weights, bias, mean, scale, tuple(high[0]))
    assert proba_low < 0.1
    assert proba_high > 0.9


def test_shipped_weights_separate_representative_frames() -> None:
    half = half_probability(
        DEFAULT_WEIGHTS,
        DEFAULT_BIAS,
        DEFAULT_MEAN,
        DEFAULT_SCALE,
        tuple(HALF_FEATURES[name] for name in FEATURES),
    )
    quarter = half_probability(
        DEFAULT_WEIGHTS,
        DEFAULT_BIAS,
        DEFAULT_MEAN,
        DEFAULT_SCALE,
        tuple(QUARTER_FEATURES[name] for name in FEATURES),
    )
    assert half > 0.9
    assert quarter < 0.1


def test_refine_only_touches_fold_states() -> None:
    for state in ("empty", "flat"):
        assert refine_fold_state(state, HALF_FEATURES) == state


def test_refine_leaves_decisions_outside_the_zone() -> None:
    # Large relative area: confidently a half crepe under the rule.
    big = _features(relative=0.8)
    assert refine_fold_state("half", big) == "half"
    # Small relative area: confidently a quarter crepe under the rule.
    small = {**_features(), **QUARTER_FEATURES, "relative": 0.1}
    assert refine_fold_state("quarter", small) == "quarter"
    assert ZONE_LOW <= 0.3 <= ZONE_HIGH


def test_refine_flips_a_boundary_zone_decision() -> None:
    # The rule calls relative 0.3 with a square silhouette a half crepe
    # (plate fraction rule); the model sees the quarter-like aspect.
    boundary = _features(
        relative=0.3,
        rect_aspect=1.15,
        plate_fraction=0.18,
        diameter_fraction=0.5,
        offset_fraction=0.15,
        solidity=0.93,
        circularity=0.65,
        edge_count=2,
    )
    assert refine_fold_state("half", boundary) == "quarter"
    # ...and the inverse when the silhouette is elongated and the model is
    # confident enough (p about 0.82).
    boundary_half = _features(relative=0.33, plate_fraction=0.30)
    assert refine_fold_state("quarter", boundary_half) == "half"


def test_low_confidence_model_output_keeps_the_rule() -> None:
    # p about 0.54: the model leans half but not beyond the flip margin.
    borderline = _features(relative=0.30, plate_fraction=0.28, rect_aspect=1.9)
    assert refine_fold_state("quarter", borderline) == "quarter"
    assert refine_fold_state("half", borderline) == "half"


@pytest.mark.parametrize("missing", list(FEATURES))
def test_missing_feature_keeps_the_rule_decision(missing: str) -> None:
    features = _features()
    del features[missing]
    assert refine_fold_state("half", features) == "half"
    assert refine_fold_state("quarter", features) == "quarter"


@pytest.mark.parametrize("name", ["relative", "rect_aspect", "solidity"])
def test_non_finite_feature_keeps_the_rule_decision(name: str) -> None:
    features = _features(**{name: math.nan})
    assert refine_fold_state("half", features) == "half"


def test_custom_weights_override_the_default_model() -> None:
    # A constant-zero model with a strong bias always votes one way.
    zero = np.zeros(len(FEATURES))
    boundary_half = _features(relative=0.33)
    assert (
        refine_fold_state("quarter", boundary_half, weights=zero, bias=-5.0)
        == "quarter"
    )
    assert (
        refine_fold_state("half", boundary_half, weights=zero, bias=-5.0)
        == "quarter"
    )
    assert (
        refine_fold_state("quarter", boundary_half, weights=zero, bias=5.0)
        == "half"
    )
    assert (
        refine_fold_state("half", boundary_half, weights=zero, bias=5.0)
        == "half"
    )
