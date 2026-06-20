"""Patch-based inference with Gaussian blending for overlap regions."""

import numpy as np
import tensorflow as tf

# Patch side must be divisible by this so the 4-level UNets (four 2x pooling
# stages) receive spatial dimensions that downsample and upsample cleanly.
_PATCH_DIVISOR = 16


def _gaussian_window(size: int, sigma: float | None = None) -> np.ndarray:
    """Generate a square 2D Gaussian weight window.

    Parameters
    ----------
    size : int
        Side length of the window in pixels.
    sigma : float or None
        Standard deviation of the Gaussian. Defaults to ``size / 6``.

    Returns
    -------
    np.ndarray
        Array of shape ``(size, size)`` with values in ``(0, 1]``,
        normalised so the peak equals 1.
    """
    if sigma is None:
        sigma = size / 6.0
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords**2) / (2 * sigma**2))
    window = np.outer(g, g)
    return (window / window.max()).astype(np.float32)


def _patch_starts(size: int, patch_size: int, stride: int) -> list[int]:
    """Compute patch start positions that cover ``[0, size)`` fully.

    Parameters
    ----------
    size : int
        Total extent along one axis.
    patch_size : int
        Size of each patch.
    stride : int
        Step between consecutive patches.

    Returns
    -------
    list[int]
        Sorted list of start positions such that every pixel is covered
        by at least one patch.
    """
    starts = list(range(0, size - patch_size + 1, stride))
    if not starts or starts[-1] + patch_size < size:
        starts.append(max(0, size - patch_size))
    return starts


def predict_with_overlap(
    model: tf.keras.Model,
    rgb_image: np.ndarray,
    patch_size: int = 256,
    stride: int | None = None,
) -> np.ndarray:
    """Run model inference on a full image using overlapping patches.

    Splits the input into overlapping square patches, runs the model on
    each patch, and blends predictions with Gaussian weights.  Gaussian
    blending eliminates the hard seam artifacts that arise from
    non-overlapping stitching.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.  Expected input ``(1, H, W, 3)``, output
        ``(1, H, W, 1)``, values in ``[0, 1]``.
    rgb_image : np.ndarray
        Input RGB image with shape ``(H, W, 3)``, values in ``[0, 1]``.
    patch_size : int
        Side length of each square patch in pixels.
    stride : int or None
        Step between consecutive patches.  Defaults to
        ``patch_size // 2`` (50 % overlap).

    Returns
    -------
    np.ndarray
        Predicted IR image with shape ``(H, W, 1)``, values in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``patch_size`` is not a multiple of 16, or if ``stride`` is
        greater than ``patch_size``.
    """
    if patch_size % _PATCH_DIVISOR != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be a multiple of "
            f"{_PATCH_DIVISOR}; the 4-level UNets pool by 2 four times and "
            "require divisible spatial dimensions."
        )
    if stride is None:
        stride = patch_size // 2
    if stride > patch_size:
        raise ValueError(f"stride ({stride}) must be ≤ patch_size ({patch_size}).")

    h, w = rgb_image.shape[:2]

    # Pad to at least patch_size in each dimension.
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    rgb_padded = np.pad(rgb_image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = rgb_padded.shape[:2]

    prediction = np.zeros((ph, pw, 1), dtype=np.float32)
    weight_map = np.zeros((ph, pw, 1), dtype=np.float32)
    window = _gaussian_window(patch_size)[..., np.newaxis]  # (P, P, 1)

    for y in _patch_starts(ph, patch_size, stride):
        for x in _patch_starts(pw, patch_size, stride):
            patch = rgb_padded[y : y + patch_size, x : x + patch_size]
            patch_t = tf.expand_dims(patch, axis=0)  # (1, P, P, 3)
            pred = model(patch_t, training=False).numpy()[0]  # (P, P, 1)
            prediction[y : y + patch_size, x : x + patch_size] += pred * window
            weight_map[y : y + patch_size, x : x + patch_size] += window

    blended = prediction / (weight_map + 1e-8)
    return blended[:h, :w]
