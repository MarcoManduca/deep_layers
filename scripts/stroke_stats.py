"""Unsupervised stroke-likeness of a signal map — no reference mask needed.

The second axis of model evaluation, independent of any ground truth including
the pseudo mask in ``scripts.pseudo_mask``. It rests on a property of the
target rather than on its location: an underdrawing is made of **strokes** —
elongated, oriented, spatially coherent — while prediction noise is isotropic
and high-frequency. Those are two statistically separable populations even
without knowing where the strokes are.

The separator is the structure tensor. Its two eigenvalues describe how
gradient energy is distributed around a pixel: one large and one small means
the neighbourhood has a dominant direction (an edge or a stroke), two similar
values mean it has none (noise, or flat). Coherence
``((l1 - l2) / (l1 + l2))^2`` turns that into ``[0, 1]``, and averaging it
weighted by gradient energy keeps flat regions — where the ratio is numerically
meaningless — from dominating the result.

Use it next to ``scripts.detection``: the two are wrong in different ways, so
agreement between them is worth much more than either alone.

Caveat worth stating out loud: this is gameable. A model with oriented
artefacts — patch-stitching seams, checkerboarding from ``Conv2DTranspose`` —
scores high while revealing nothing. Never read it without looking at the maps.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

_EPS = 1e-12


@dataclass(frozen=True)
class StrokeStats:
    """How stroke-like a signal map is.

    Attributes
    ----------
    coherence : float
        Energy-weighted mean coherence in ``[0, 1]``. Near ``1`` the signal is
        dominated by oriented, line-like structure; near ``0`` it is isotropic,
        i.e. noise. A straight stroke scores ~``0.94``, uniform noise ~``0.08``.
    gradient_energy : float
        Mean total gradient energy (``l1 + l2``). The weight behind the
        coherence figure: a very flat map can report a coherence value that
        rests on almost no signal, and this is how you notice.
    coherence_map : np.ndarray
        Per-pixel coherence, for display alongside the signal it describes.
    """

    coherence: float
    gradient_energy: float
    coherence_map: np.ndarray = field(repr=False)


def stroke_coherence(signal: np.ndarray, sigma: float = 2.0) -> StrokeStats:
    """Measure how much of a signal map is oriented, stroke-like structure.

    Parameters
    ----------
    signal : np.ndarray
        Signal map of shape ``(H, W)`` or ``(H, W, 1)`` — a delta, a
        ``|z|``-score, or any other candidate. Only its spatial structure is
        used, so the absolute scale is irrelevant and signals on different
        scales stay comparable.
    sigma : float
        Standard deviation of the Gaussian used to smooth the structure
        tensor. Sets the scale of the strokes being looked for: larger values
        respond to broader marks and suppress fine noise.

    Returns
    -------
    StrokeStats
    """
    values = signal.squeeze().astype(np.float64)

    elements = structure_tensor(values, sigma=sigma, order="rc")
    eigenvalues = structure_tensor_eigenvalues(elements)
    major, minor = eigenvalues[0], eigenvalues[1]

    energy = major + minor
    coherence_map = ((major - minor) / (energy + _EPS)) ** 2

    total = float(energy.sum())
    weighted = float((coherence_map * energy).sum() / (total + _EPS))

    return StrokeStats(
        coherence=weighted,
        gradient_energy=float(energy.mean()),
        coherence_map=coherence_map,
    )


def rank_by_coherence(
    signals: dict[str, np.ndarray], sigma: float = 2.0
) -> dict[str, StrokeStats]:
    """Score several signal maps by stroke-likeness, best first.

    Parameters
    ----------
    signals : dict[str, np.ndarray]
        Maps a signal name to its map.
    sigma : float
        Structure-tensor smoothing, applied identically to every signal.

    Returns
    -------
    dict[str, StrokeStats]
        Results under the same keys, ordered most stroke-like first.
    """
    results = {name: stroke_coherence(s, sigma=sigma) for name, s in signals.items()}
    return dict(sorted(results.items(), key=lambda kv: kv[1].coherence, reverse=True))
