"""Optional logistic refinement for the half/quarter boundary zone.

The fold-state rule in :mod:`maus_export.fold_state` is a few thresholds
on the filled area, the plate coverage and the minimum-rectangle aspect.
It is interpretable and stable, but the two thresholds it relies on most
(relative filled area and plate fraction) overlap between a half crepe and
a quarter crepe. In that boundary zone a small logistic model trained on
the geometry features of the local export separates the states better
(10-fold cross validation on the 397 half/quarter check frames: rule
96.5%, model 98.5%, mostly better quarter recall).

The model is deliberately tiny and dependency-free (numpy only) so it can
run inside the 200 ms/frame budget and be replaced later by an ONNX model
without touching the rest of the pipeline. :func:`refine_fold_state` only
acts when the rule already says ``half`` or ``quarter`` *and* the relative
area lies in the boundary zone; everywhere else the rule's decision stands
unchanged.

The shipped weights are fit on the same local export the rule thresholds
were tuned on, so the cross-validation number above is the honest
generalisation estimate, not a held-out one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

#: Geometry features the model consumes, in order.
FEATURES = (
    "relative",
    "plate_fraction",
    "rect_aspect",
    "diameter_fraction",
    "offset_fraction",
    "hole_fraction",
    "solidity",
    "circularity",
    "edge_count",
)

#: Relative filled-area window in which the rule is uncertain.
ZONE_LOW = 0.24
ZONE_HIGH = 0.42

#: Minimum distance of the model output from an even split for a flip.
FLIP_MARGIN = 0.25

# Fit on the local export (see the module docstring for the caveat).
DEFAULT_MEAN = np.array(
    [
        0.351062,
        0.245332,
        1.753360,
        0.480096,
        0.269573,
        0.014118,
        0.928860,
        0.496592,
        1.229219,
    ]
)
DEFAULT_SCALE = np.array(
    [
        0.136838,
        0.086799,
        0.565649,
        0.096196,
        0.064046,
        0.023703,
        0.074262,
        0.126537,
        0.834008,
    ]
)
DEFAULT_WEIGHTS = np.array(
    [
        2.547125,
        0.684202,
        3.634169,
        -0.044355,
        -0.623608,
        0.706435,
        -0.580712,
        0.161275,
        -0.468180,
    ]
)
DEFAULT_BIAS = -0.368402


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.5,
    iterations: int = 6000,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a standardised logistic regression by gradient descent.

    Returns ``(weights, bias, mean, scale)``; feed all four back into
    :func:`half_probability`. Used to retrain the shipped model on new
    data; the pipeline itself only needs the inference helpers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1])
    bias = 0.0
    count = len(z)
    for _ in range(iterations):
        proba = 1.0 / (1.0 + np.exp(-(z @ weights + bias)))
        delta = proba - y
        weights -= learning_rate * (z.T @ delta / count + l2 * weights)
        bias -= learning_rate * delta.mean()
    return weights, float(bias), mean, scale


def half_probability(
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
    values: tuple[float, ...],
) -> float:
    """Probability that the frame is a half crepe (model output)."""
    z = (
        np.asarray(values, dtype=float) - np.asarray(mean, dtype=float)
    ) / np.asarray(scale, dtype=float)
    return float(1.0 / (1.0 + math.exp(-(float(z @ weights) + bias))))


def refine_fold_state(
    state: str,
    features: Mapping[str, Any],
    *,
    weights: np.ndarray = DEFAULT_WEIGHTS,
    bias: float = DEFAULT_BIAS,
    mean: np.ndarray = DEFAULT_MEAN,
    scale: np.ndarray = DEFAULT_SCALE,
) -> str:
    """Let the boundary-zone model second-guess a rule decision.

    Only ``half`` and ``quarter`` states with a relative filled area in
    ``[ZONE_LOW, ZONE_HIGH]`` are reconsidered; any missing or
    non-finite feature leaves the rule's decision untouched. A flip also
    requires the model to be at least :data:`FLIP_MARGIN` away from an even
    split, so borderline frames keep the rule's decision. On the local export
    every correct flip in the zone clears that margin, so it costs nothing
    offline while rejecting low-confidence flips on noisy live frames.
    """
    margin = FLIP_MARGIN
    if state not in ("half", "quarter"):
        return state
    try:
        values = tuple(float(features[name]) for name in FEATURES)
    except (KeyError, TypeError, ValueError):
        return state
    if any(math.isnan(value) for value in values):
        return state
    if not ZONE_LOW <= values[0] <= ZONE_HIGH:
        return state
    proba = half_probability(weights, bias, mean, scale, values)
    if state == "half" and proba < 0.5 - margin:
        return "quarter"
    if state == "quarter" and proba > 0.5 + margin:
        return "half"
    return state
