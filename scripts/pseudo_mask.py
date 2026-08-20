"""Model-independent pseudo ground truth for hidden detail in an RGB/IR pair.

Scoring which model best reveals underdrawings needs a reference, and the only
annotated one available (``modern``, a purpose-built test pair) is a single
image whose hidden details were placed by someone who knows what the pipeline
looks for — no statistical power, and a materially different object from an
aged painting. This module derives a reference from the data instead, so every
real artwork in the test fold can serve as evidence.

The operational definition of "hidden detail" needs no human: **structure that
is present in the IR and not explained by the RGB**. Both halves are already
computable with ``scripts.delta_analysis.compute_local_stats``, run
cross-modally on ``(real_IR, grayscale RGB)`` rather than on
``(real_IR, predicted_IR)``:

- ``var_real`` — how much local structure the IR carries at all;
- ``structure`` — how much of it the RGB accounts for (the local correlation
  between the two modalities).

The score is high where the first is high and the second is low.

Two properties make this a fair referee: it is computed **only from the data**,
so it cannot favour any architecture, and it applies to every paired image,
turning a sample of one into a sample of twenty-four.

Two limits, both important enough to state before the first result is read.

**It marks a superset of what a conservator cares about.** Canvas weave, wood
grain, fillers and old restorations are all IR structure the RGB cannot
explain, and all of it scores here. Read the output as a relative ranking
between candidate signals ("model A surfaces more IR-only structure than model
B"), never as an absolute measure of "found the underdrawing".

**It is a fair referee across models, not across signal types.** Nothing here
knows which architecture produced what, so comparing models is sound. But the
score shares its mathematical form with
``delta_analysis``'s structural delta — both are built on the same local
windowed structure statistic — so a structural signal is favoured against, say,
a raw delta *by construction*, not on merit. Compare like with like: rank
architectures within one signal type, and use
``scripts.stroke_stats`` (which shares nothing with this construction) as the
independent second opinion whenever signal types must be compared.
"""

from dataclasses import dataclass, field

import numpy as np

from scripts.delta_analysis import compute_local_stats

_EPS = 1e-8

# Absolute floor on the local IR standard deviation treated as real structure,
# in [0, 1] image units: below this, the variation is under a quarter of one
# 8-bit gray level and is measurement/quantisation noise, not detail.
_MIN_CONTRAST = 1e-3

# ITU-R BT.601 luma coefficients, matching the grayscale conversion PIL applies
# when the IR images are loaded with ``.convert("L")``.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to luma, matching how the IR images are loaded.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image of shape ``(H, W, 3)``, values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Grayscale image of shape ``(H, W)``.
    """
    return np.tensordot(
        rgb.astype(np.float32), np.array(_LUMA_WEIGHTS), axes=([-1], [0])
    )


@dataclass(frozen=True)
class PseudoMaskResult:
    """Cross-modal pseudo ground truth for one RGB/IR pair.

    Attributes
    ----------
    score : np.ndarray
        Continuous ``[0, 1]`` map, high where the IR carries structure the RGB
        does not explain. This is the reference a candidate signal is scored
        against.
    ir_contrast : np.ndarray
        Local IR standard deviation, normalised to ``[0, 1]``. The "is there
        any structure here at all" factor.
    cross_structure : np.ndarray
        ``|structure(IR, RGB)|`` in ``[0, 1]``. The "is it explained by the
        RGB" factor: ``1`` where the two modalities agree locally.
    """

    score: np.ndarray = field(repr=False)
    ir_contrast: np.ndarray = field(repr=False)
    cross_structure: np.ndarray = field(repr=False)

    def binarize(self, percentile: float = 95.0) -> np.ndarray:
        """Threshold the score into a binary mask at a percentile.

        Detection metrics need a binary reference; where to cut is a choice
        about how much of the image counts as "detail", so it stays a
        parameter rather than a hard-coded constant.

        Parameters
        ----------
        percentile : float
            Percentile of ``score`` used as the threshold. ``95.0`` marks the
            top 5% of pixels as positive.

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(H, W)``.

        Raises
        ------
        ValueError
            If ``percentile`` is outside ``[0, 100)``.
        """
        if not 0 <= percentile < 100:
            raise ValueError(f"percentile must be in [0, 100), got {percentile}.")
        return self.score > float(np.percentile(self.score, percentile))


def cross_modal_pseudo_mask(
    rgb: np.ndarray,
    real_ir: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    contrast_percentile: float = 99.0,
) -> PseudoMaskResult:
    """Score every pixel by how much unexplained IR structure it carries.

    ``score = normalised local IR contrast * (1 - |local correlation(IR, RGB)|)``

    The **absolute value** matters: a region where the IR is strongly
    *anti*-correlated with the RGB (dark paint reading bright in IR, which is
    common) is perfectly explained by the RGB — just with inverted sign. Using
    the signed correlation would flag every such region as hidden detail, which
    is the single easiest way to get this wrong.

    The correlation is computed directly from ``compute_local_stats`` rather
    than reusing ``compute_ssim_components``'s ``structure`` term, whose ``c3``
    stabiliser biases this specific case: ``c3`` adds to numerator and
    denominator with the same sign, so it leaves a perfect positive correlation
    at exactly ``1`` while pulling a perfect *negative* one towards ``0``. On
    low-contrast texture that turns "fully explained, inverted" into "looks
    unexplained". Numerical safety in near-flat regions is instead left to the
    ``ir_contrast`` factor, which is what should be suppressing them anyway.

    The window defaults match ``tf.image.ssim`` and
    ``scripts.delta_analysis``, so this map is measured on the same spatial
    scale as every other signal in the pipeline.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image of shape ``(H, W, 3)``, values in ``[0, 1]``.
    real_ir : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``, values in
        ``[0, 1]``.
    window_size : int
        Side length of the Gaussian window for the local statistics.
    sigma : float
        Standard deviation of that window.
    contrast_percentile : float
        Percentile of the local IR contrast mapped to ``1.0`` before clipping.
        Using a high percentile rather than the maximum keeps a handful of
        extreme pixels from compressing the whole map towards zero.

    Returns
    -------
    PseudoMaskResult

    Raises
    ------
    ValueError
        If the RGB and IR spatial dimensions differ.
    """
    ir = real_ir.squeeze().astype(np.float32)
    gray = to_grayscale(rgb)

    if gray.shape != ir.shape:
        raise ValueError(
            f"RGB and IR must have the same spatial shape, got {gray.shape} "
            f"and {ir.shape}."
        )

    stats = compute_local_stats(ir, gray, window_size=window_size, sigma=sigma)

    ir_std = np.sqrt(np.maximum(stats.var_real, 0.0))
    rgb_std = np.sqrt(np.maximum(stats.var_pred, 0.0))
    correlation = stats.cov / (ir_std * rgb_std + _EPS)
    cross_structure = np.clip(np.abs(correlation), 0.0, 1.0)

    # The floor matters: a percentile alone is a relative scale, so on a uniform
    # IR region it would rescale pure floating-point noise up to 1.0 and report
    # a flat area as maximal hidden detail. Below _MIN_CONTRAST the local
    # variation is under a quarter of one 8-bit gray level — not structure.
    reference = max(float(np.percentile(ir_std, contrast_percentile)), _MIN_CONTRAST)
    ir_contrast = np.clip(ir_std / reference, 0.0, 1.0)

    return PseudoMaskResult(
        score=ir_contrast * (1.0 - cross_structure),
        ir_contrast=ir_contrast,
        cross_structure=cross_structure,
    )
