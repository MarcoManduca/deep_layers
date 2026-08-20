"""Detection scores for a candidate hidden-detail signal against a reference mask.

Turns "this delta map looks better" into a number. Every signal the pipeline
produces — raw delta, structural delta, fixed-window normalized delta, learned
z-score, per architecture and per loss variant — is a per-pixel ranking of "how
likely is this pixel to hold hidden detail", which is exactly what a detection
metric scores.

Pair with ``scripts.pseudo_mask`` for a reference derived from the data itself,
or with a hand-annotated mask when one exists. The metrics are indifferent to
where the mask came from.

``roc_auc`` answers "does this signal rank detail above non-detail?" and is
insensitive to how much of the image is positive. ``average_precision`` answers
"if a conservator follows the top of this ranking, how much of what they find
is real?" and *is* prevalence-dependent — which is why ``prevalence`` is
reported alongside it and ``lift`` normalises it against chance. Comparing
average precision across images with different mask densities without looking
at prevalence is meaningless.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class DetectionResult:
    """How well one signal map ranks the pixels of a reference mask.

    Attributes
    ----------
    auroc : float
        Area under the ROC curve. ``0.5`` is chance, ``1.0`` perfect, below
        ``0.5`` means the signal ranks detail *below* background.
    average_precision : float
        Area under the precision-recall curve. Chance level equals
        ``prevalence``, not ``0.5``.
    prevalence : float
        Fraction of pixels marked positive in the reference mask.
    """

    auroc: float
    average_precision: float
    prevalence: float

    @property
    def lift(self) -> float:
        """Average precision relative to chance — ``1.0`` means no better."""
        return self.average_precision / (self.prevalence + 1e-12)


def evaluate_detection(
    signal: np.ndarray,
    mask: np.ndarray,
    max_samples: int = 2_000_000,
    seed: int = 0,
) -> DetectionResult:
    """Score how well a signal map ranks the pixels of a reference mask.

    Parameters
    ----------
    signal : np.ndarray
        Candidate signal, shape ``(H, W)`` or ``(H, W, 1)``. **Must be a
        magnitude**: higher = more likely to be hidden detail. Pass ``abs(z)``
        rather than a signed z-score, otherwise dark anomalies count against
        the signal.
    mask : np.ndarray
        Reference mask of the same spatial shape, boolean or ``0``/``1``.
    max_samples : int
        Pixels above this count are randomly subsampled before scoring.
    seed : int
        Seed for that subsampling, so the score is reproducible.

    Returns
    -------
    DetectionResult

    Raises
    ------
    ValueError
        If the shapes differ, or the mask is entirely positive or entirely
        negative (no ranking to score).
    """
    values = signal.squeeze().astype(np.float64).ravel()
    labels = mask.squeeze().astype(bool).ravel()

    if values.shape != labels.shape:
        raise ValueError(
            f"signal and mask must have the same shape, got {signal.squeeze().shape} "
            f"and {mask.squeeze().shape}."
        )

    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError(
            "mask must contain both positive and negative pixels; got "
            f"{positives} positives out of {labels.size}."
        )

    prevalence = positives / labels.size

    if values.size > max_samples:
        rng = np.random.default_rng(seed)
        index = rng.choice(values.size, size=max_samples, replace=False)
        values, labels = values[index], labels[index]

    return DetectionResult(
        auroc=float(roc_auc_score(labels, values)),
        average_precision=float(average_precision_score(labels, values)),
        prevalence=float(prevalence),
    )


def rank_signals(
    signals: dict[str, np.ndarray],
    mask: np.ndarray,
    max_samples: int = 2_000_000,
    seed: int = 0,
) -> dict[str, DetectionResult]:
    """Score several candidate signals against one reference mask.

    The cross-model, cross-signal comparison: one mask, one table, every
    candidate scored on identical terms.

    Parameters
    ----------
    signals : dict[str, np.ndarray]
        Maps a signal name (e.g. ``"unet_nll (beta) |z|"``) to its magnitude
        map.
    mask : np.ndarray
        Reference mask shared by all of them.
    max_samples : int
        Subsampling cap, applied identically to every signal.
    seed : int
        Subsampling seed — the same for every signal, so they are all scored
        on the same pixels.

    Returns
    -------
    dict[str, DetectionResult]
        Results under the same keys, ordered best-AUROC first.
    """
    results = {
        name: evaluate_detection(signal, mask, max_samples=max_samples, seed=seed)
        for name, signal in signals.items()
    }
    return dict(sorted(results.items(), key=lambda kv: kv[1].auroc, reverse=True))
