"""Confidence calibration via Platt scaling.

The verify gate works in raw cosine-distance space. A distance of 0.28
means little on its own — is that a confident match or a borderline one?
Platt scaling fits a sigmoid that maps distance → a calibrated
probability that the utterance is the *same* speaker:

    P(same | d) = 1 / (1 + exp(A·d + B))

Because lower distance means "more likely the same speaker", the fit
yields A > 0, so the probability falls smoothly as distance grows.

Training pairs come from the enrolled samples themselves (see
``SpeakerStore.collect_calibration_data``):

  * **genuine** — each sample vs. its own speaker's *leave-one-out*
    centroid (label 1). Leave-one-out keeps the genuine distances honest;
    including the sample in its own centroid would make them
    artificially small.
  * **impostor** — each sample vs. every *other* speaker's centroid
    (label 0).

This mirrors the verify path (sample-embedding vs. centroid) so the
calibration is in the same space the gate actually scores in. Once
recognition events accumulate trustworthy labels they can be folded into
the same fit — the API is just (distances, labels).

The fit is the canonical, numerically-stable routine from Lin, Lin &
Weng (2007), "A Note on Platt's Probabilistic Outputs for Support Vector
Machines" — the same one libsvm uses.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

_LOGGER = logging.getLogger("murdock.calibration")


def fit_platt(
    distances: Sequence[float],
    labels: Sequence[int],
    *,
    max_iter: int = 100,
    min_step: float = 1e-10,
    sigma: float = 1e-12,
    eps: float = 1e-5,
) -> Optional[tuple[float, float]]:
    """Fit sigmoid parameters (A, B) for ``P = 1/(1+exp(A·d+B))``.

    Returns ``None`` when the data has only one class (no genuine or no
    impostor pairs), in which case calibration isn't possible.
    """
    dec = np.asarray(distances, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if dec.size == 0:
        return None

    prior1 = float(np.sum(y > 0))   # genuine count
    prior0 = float(dec.size - prior1)  # impostor count
    if prior1 == 0 or prior0 == 0:
        return None

    # Platt's target smoothing (avoids 0/1 targets → infinite logits).
    hi = (prior1 + 1.0) / (prior1 + 2.0)
    lo = 1.0 / (prior0 + 2.0)
    t = np.where(y > 0, hi, lo)

    A = 0.0
    B = math.log((prior0 + 1.0) / (prior1 + 1.0))

    def objective(a: float, b: float) -> float:
        fApB = dec * a + b
        pos = fApB >= 0
        out = np.empty_like(fApB)
        out[pos] = t[pos] * fApB[pos] + np.log1p(np.exp(-fApB[pos]))
        out[~pos] = (t[~pos] - 1.0) * fApB[~pos] + np.log1p(np.exp(fApB[~pos]))
        return float(np.sum(out))

    fval = objective(A, B)

    for _ in range(max_iter):
        fApB = dec * A + B
        pos = fApB >= 0
        p = np.empty_like(fApB)
        q = np.empty_like(fApB)
        ep = np.exp(-fApB[pos])
        p[pos] = ep / (1.0 + ep)
        q[pos] = 1.0 / (1.0 + ep)
        en = np.exp(fApB[~pos])
        p[~pos] = 1.0 / (1.0 + en)
        q[~pos] = en / (1.0 + en)

        d2 = p * q
        h11 = float(np.sum(dec * dec * d2)) + sigma
        h22 = float(np.sum(d2)) + sigma
        h21 = float(np.sum(dec * d2))
        d1 = t - p
        g1 = float(np.sum(dec * d1))
        g2 = float(np.sum(d1))

        if abs(g1) < eps and abs(g2) < eps:
            break

        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        stepsize = 1.0
        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = objective(newA, newB)
            if newf < fval + 1e-4 * stepsize * gd:
                A, B, fval = newA, newB, newf
                break
            stepsize /= 2.0
        if stepsize < min_step:
            # Line search failed — converged as far as we can.
            break

    return A, B


@dataclass
class Calibrator:
    """Holds fitted Platt parameters and turns a distance into P(same)."""

    a: float = 0.0
    b: float = 0.0
    fitted: bool = False
    n_genuine: int = 0
    n_impostor: int = 0
    fitted_at: float = 0.0

    def probability(self, distance: float) -> Optional[float]:
        """Calibrated P(same speaker) for a cosine distance, or None."""
        if not self.fitted:
            return None
        z = self.a * float(distance) + self.b
        # Numerically stable logistic for 1/(1+exp(z)).
        if z >= 0:
            ez = math.exp(-z)
            prob = ez / (1.0 + ez)
        else:
            prob = 1.0 / (1.0 + math.exp(z))
        return max(0.0, min(1.0, prob))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Calibrator":
        if not data:
            return cls()
        return cls(
            a=float(data.get("a", 0.0)),
            b=float(data.get("b", 0.0)),
            fitted=bool(data.get("fitted", False)),
            n_genuine=int(data.get("n_genuine", 0)),
            n_impostor=int(data.get("n_impostor", 0)),
            fitted_at=float(data.get("fitted_at", 0.0)),
        )


def compute_adaptive_thresholds(
    per_speaker: dict,
    global_threshold: float,
    *,
    max_delta: float = 0.08,
    min_genuine: int = 4,
    min_impostor: int = 4,
) -> dict:
    """Derive a per-speaker verify threshold from their score distributions.

    For each speaker with enough data, the threshold is the midpoint
    between their genuine 95th percentile (their own voice on a bad day)
    and their impostor 5th percentile (the closest a stranger gets to
    their profile) — i.e. the per-speaker version of the global
    threshold-recommendation logic, and (via the fitted sigmoid) a
    per-speaker probability cut.

    Deliberately bounded: the result is clamped to
    ``global_threshold ± max_delta`` so a small enrollment can nudge a
    speaker's gate, never swing it wide open. Speakers whose
    distributions overlap (or with too little data) get no entry and
    keep the global threshold.
    """
    out: dict = {}
    for name, stats in (per_speaker or {}).items():
        genuine = stats.get("genuine") or []
        impostor = stats.get("impostor") or []
        if len(genuine) < min_genuine or len(impostor) < min_impostor:
            continue
        g95 = float(np.percentile(np.asarray(genuine, dtype=np.float64), 95))
        i05 = float(np.percentile(np.asarray(impostor, dtype=np.float64), 5))
        if i05 - g95 <= 0:
            # Overlapping distributions: the data can't justify a custom
            # gate — more/better samples needed, not a different number.
            continue
        mid = (g95 + i05) / 2.0
        lo = max(0.05, global_threshold - max_delta)
        hi = global_threshold + max_delta
        out[name] = round(min(hi, max(lo, mid)), 4)
    return out


def calibrator_from_pairs(
    distances: Sequence[float], labels: Sequence[int]
) -> Calibrator:
    """Fit a :class:`Calibrator` from labeled distance pairs.

    Returns an unfitted calibrator (``fitted=False``) when the data can't
    support a fit, so the caller transparently falls back to the raw
    ``1 - distance`` confidence.
    """
    n_genuine = int(sum(1 for v in labels if v > 0))
    n_impostor = int(len(labels) - n_genuine)
    params = fit_platt(distances, labels)
    if params is None:
        _LOGGER.info(
            "Calibration skipped — need both genuine and impostor pairs "
            "(genuine=%d, impostor=%d)", n_genuine, n_impostor,
        )
        return Calibrator(n_genuine=n_genuine, n_impostor=n_impostor)
    a, b = params
    _LOGGER.info(
        "Calibration fitted: A=%.4f B=%.4f (genuine=%d, impostor=%d)",
        a, b, n_genuine, n_impostor,
    )
    return Calibrator(
        a=a, b=b, fitted=True,
        n_genuine=n_genuine, n_impostor=n_impostor,
        fitted_at=time.time(),
    )
