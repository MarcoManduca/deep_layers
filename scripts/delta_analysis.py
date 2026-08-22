"""Regional analysis of the IR delta, decomposing it into substrate/acquisition
effects (luminance, contrast) versus genuine structural discontinuities.

Takes the real and predicted IR images and produces complementary maps that
help separate a gray-level shift (substrate, illumination, exposure) from an
actual hidden mark (underdrawing, pentimento, reused support). See
``note.md`` for the design rationale.
"""

from dataclasses import dataclass

import numpy as np

_EPS = 1e-8


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_local_filter(x: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian filter with ``reflect`` padding.

    Public so other modules can smooth a map with the exact same window used
    here (e.g. ``scripts.calibration.structural_zscore`` smoothing ``sigma``
    to match ``structural_delta``'s window before combining them).

    Parameters
    ----------
    x : np.ndarray
        Input image, shape ``(H, W)``.
    size : int
        Side length of the (square) Gaussian window.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        Filtered image, shape ``(H, W)``.
    """
    kernel = _gaussian_kernel_1d(size, sigma)
    pad = size // 2
    padded = np.pad(x, pad, mode="reflect")

    def _convolve_valid(v: np.ndarray) -> np.ndarray:
        return np.convolve(v, kernel, mode="valid")

    rows = np.apply_along_axis(_convolve_valid, 1, padded)
    return np.apply_along_axis(_convolve_valid, 0, rows)


@dataclass
class LocalStats:
    """Local (windowed) first- and second-order statistics of an image pair.

    Attributes
    ----------
    mu_real, mu_pred : np.ndarray
        Local means, shape ``(H, W)``.
    var_real, var_pred : np.ndarray
        Local variances, shape ``(H, W)``.
    cov : np.ndarray
        Local covariance between ``real`` and ``pred``, shape ``(H, W)``.
    """

    mu_real: np.ndarray
    mu_pred: np.ndarray
    var_real: np.ndarray
    var_pred: np.ndarray
    cov: np.ndarray


def compute_local_stats(
    real: np.ndarray,
    pred: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
) -> LocalStats:
    """Compute local Gaussian-windowed statistics for an IR image pair.

    Same window defaults (``11x11``, ``sigma=1.5``) as ``tf.image.ssim``,
    so the resulting luminance/contrast/structure maps are consistent with
    the scalar SSIM already used elsewhere in the project.

    Parameters
    ----------
    real : np.ndarray
        Ground-truth IR image, shape ``(H, W)``, values in ``[0, 1]``.
    pred : np.ndarray
        Predicted IR image, shape ``(H, W)``, values in ``[0, 1]``.
    window_size : int
        Side length of the square Gaussian window.
    sigma : float
        Standard deviation of the Gaussian window.

    Returns
    -------
    LocalStats
    """
    mu_real = gaussian_local_filter(real, window_size, sigma)
    mu_pred = gaussian_local_filter(pred, window_size, sigma)

    var_real = gaussian_local_filter(real * real, window_size, sigma) - mu_real**2
    var_pred = gaussian_local_filter(pred * pred, window_size, sigma) - mu_pred**2
    cov = gaussian_local_filter(real * pred, window_size, sigma) - mu_real * mu_pred

    return LocalStats(mu_real, mu_pred, var_real, var_pred, cov)


@dataclass
class SSIMComponents:
    """Per-pixel luminance, contrast and structure maps (Wang et al. SSIM).

    Attributes
    ----------
    luminance, contrast, structure : np.ndarray
        Maps of shape ``(H, W)``, each in (approximately) ``[-1, 1]``.
    """

    luminance: np.ndarray
    contrast: np.ndarray
    structure: np.ndarray


def compute_ssim_components(
    stats: LocalStats,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> SSIMComponents:
    """Decompose local SSIM into its luminance, contrast and structure maps.

    SSIM is the product ``luminance * contrast * structure``; this function
    exposes the three terms separately instead of the aggregated scalar
    index. Per the design proposal, the ``structure`` map is the "hidden
    detail" indicator, while luminance/contrast differences are treated as
    plausible substrate/acquisition effects.

    Parameters
    ----------
    stats : LocalStats
        Local statistics from ``compute_local_stats``.
    max_val : float
        Dynamic range of the input images (``1.0`` for ``[0, 1]`` images).
    k1, k2 : float
        SSIM stabilization constants (same defaults as ``tf.image.ssim``).

    Returns
    -------
    SSIMComponents
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    c3 = c2 / 2

    std_real = np.sqrt(np.maximum(stats.var_real, 0.0))
    std_pred = np.sqrt(np.maximum(stats.var_pred, 0.0))

    luminance = (2 * stats.mu_real * stats.mu_pred + c1) / (
        stats.mu_real**2 + stats.mu_pred**2 + c1
    )
    contrast = (2 * std_real * std_pred + c2) / (stats.var_real + stats.var_pred + c2)
    structure = (stats.cov + c3) / (std_real * std_pred + c3)

    return SSIMComponents(luminance, contrast, structure)


def _min_max_normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + _EPS)


@dataclass
class DeltaAnalysisResult:
    """Complementary delta maps produced by ``analyze_delta``.

    Attributes
    ----------
    raw_delta : np.ndarray
        Pixel-wise absolute delta ``|real - pred|`` (the current baseline).
    luminance_map, contrast_map, structure_map : np.ndarray
        Local SSIM components (see ``compute_ssim_components``).
    structural_delta : np.ndarray
        ``1 - structure_map``; the "hidden detail" indicator.
    confidence_map : np.ndarray
        Agreement between ``raw_delta`` and ``structural_delta`` after
        min-max normalization: high where both techniques agree the region
        stands out, low where they diverge.
    """

    raw_delta: np.ndarray
    luminance_map: np.ndarray
    contrast_map: np.ndarray
    structure_map: np.ndarray
    structural_delta: np.ndarray
    confidence_map: np.ndarray


def analyze_delta(
    real_ir: np.ndarray,
    pred_ir: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
) -> DeltaAnalysisResult:
    """Produce complementary regional analysis maps for an IR image pair.

    Decomposes local SSIM into luminance/contrast/structure to separate
    genuine hidden-detail signal from substrate/acquisition-driven
    gray-level shifts. See ``note.md`` for the full rationale.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``, values in
        ``[0, 1]``.
    pred_ir : np.ndarray
        Predicted IR image, same shape convention as ``real_ir``.
    window_size : int
        Side length of the Gaussian window used for the SSIM decomposition.
    sigma : float
        Standard deviation of the Gaussian window.

    Returns
    -------
    DeltaAnalysisResult
    """
    real = real_ir.squeeze().astype(np.float32)
    pred = pred_ir.squeeze().astype(np.float32)

    raw_delta = np.abs(real - pred)

    stats = compute_local_stats(real, pred, window_size=window_size, sigma=sigma)
    components = compute_ssim_components(stats)
    structural_delta = 1.0 - components.structure

    raw_norm = _min_max_normalize(raw_delta)
    structural_norm = _min_max_normalize(structural_delta)
    confidence_map = 1.0 - np.abs(raw_norm - structural_norm)

    return DeltaAnalysisResult(
        raw_delta=raw_delta,
        luminance_map=components.luminance,
        contrast_map=components.contrast,
        structure_map=components.structure,
        structural_delta=structural_delta,
        confidence_map=confidence_map,
    )
