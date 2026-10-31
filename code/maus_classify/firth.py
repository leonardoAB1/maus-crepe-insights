"""Firth-penalized logistic regression with profile likelihood intervals.

Plain maximum likelihood is biased away from zero, and Wald intervals are too
narrow, when events are few per parameter (here as few as 9 events). Firth's
penalty (Jeffreys prior, half the log determinant of the Fisher information)
removes the first-order bias and stays finite under separation. Intervals and
p-values come from the penalized profile likelihood, as in R's logistf
(Heinze and Schemper 2002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2

TOL = 1e-8
MAX_ITER = 200
MAX_STEP = 5.0  # cap on one Newton step in log odds, as in logistf
MAX_HALVINGS = 30
CI_LEVEL = 0.95
BRACKET_START = 1.0
BRACKET_LIMIT = 50.0  # log odds; a bound beyond this is reported as inf


def _loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Penalized log likelihood: log L + 0.5 log det I."""
    eta = X @ beta
    p = expit(eta)
    w = p * (1 - p)
    info = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return -math.inf
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    return ll + 0.5 * logdet


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    fixed: int | None = None,
    value: float = 0.0,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Maximise the penalized likelihood, optionally with beta[fixed] held at
    `value`. Returns (beta, penalized log likelihood)."""
    k = X.shape[1]
    beta = np.zeros(k) if start is None else start.astype(float).copy()
    free = np.ones(k, bool)
    if fixed is not None:
        beta[fixed] = value
        free[fixed] = False
    pl = _loglik(X, y, beta)
    for _ in range(MAX_ITER):
        p = expit(X @ beta)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        score = X.T @ (y - p + h * (0.5 - p))
        step = np.zeros(k)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        biggest = np.max(np.abs(step))
        if biggest > MAX_STEP:
            step *= MAX_STEP / biggest
        for _ in range(MAX_HALVINGS):
            new = beta + step
            new_pl = _loglik(X, y, new)
            if new_pl >= pl - TOL:
                break
            step /= 2
        beta, pl = new, new_pl
        if np.max(np.abs(step)) < TOL:
            break
    return beta, pl


@dataclass
class FirthResult:
    beta: np.ndarray
    loglik: float
    X: np.ndarray
    y: np.ndarray

    def _drop(self, j: int, value: float) -> float:
        """Twice the penalized log likelihood drop with beta[j] = value."""
        _, pl = _fit(self.X, self.y, fixed=j, value=value, start=self.beta)
        return 2 * (self.loglik - pl)

    def p_value(self, j: int) -> float:
        """Penalized likelihood ratio test of beta[j] = 0."""
        return float(chi2.sf(max(self._drop(j, 0.0), 0.0), df=1))

    def profile_ci(self, j: int, level: float = CI_LEVEL) -> list[float]:
        """Profile penalized likelihood interval for beta[j] (log odds)."""
        crit = chi2.ppf(level, df=1)
        b = float(self.beta[j])
        out = []
        for sign in (-1.0, 1.0):
            width = BRACKET_START
            while self._drop(j, b + sign * width) < crit:
                width *= 2
                if width > BRACKET_LIMIT:
                    break
            if width > BRACKET_LIMIT:
                out.append(sign * math.inf)
                continue
            ends = sorted((b, b + sign * width))
            root = brentq(lambda t: self._drop(j, t) - crit, *ends, xtol=1e-6)
            out.append(float(root))
        return out


def firth_logit(X: np.ndarray, y: np.ndarray) -> FirthResult:
    """Fit Firth logistic regression. `X` must include an intercept."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, pl = _fit(X, y)
    return FirthResult(beta=beta, loglik=pl, X=X, y=y)
