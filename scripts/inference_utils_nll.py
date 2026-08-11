"""Patch-based inference with Gaussian blending for 2-channel NLL models.

Generalises ``scripts.inference_utils.predict_with_overlap`` (kept
untouched) to a heteroscedastic model whose output carries two channels —
``mu`` and ``log_var`` — instead of one, reusing its private Gaussian
window / patch-tiling helpers to avoid duplicating that logic.
"""

import numpy as np
import tensorflow as tf

from scripts.inference_utils import _PATCH_DIVISOR, _gaussian_window, _patch_starts


def predict_with_overlap_nll(
    model: tf.keras.Model,
    rgb_image: np.ndarray,
    patch_size: int = 256,
    stride: int | None = None,
) -> np.ndarray:
    """Run heteroscedastic-model inference on a full image using overlapping patches.

    Splits the input into overlapping square patches, runs the model on
    each patch, and blends predictions with Gaussian weights — identical
    stitching strategy to
    :func:`scripts.inference_utils.predict_with_overlap`, applied
    independently to both output channels.

    Parameters
    ----------
    model : tf.keras.Model
        Trained heteroscedastic model. Expected input ``(1, H, W, 3)``,
        output ``(1, H, W, 2)`` — ``mu`` (values in ``[0, 1]``) in channel
        0, ``log_var`` in channel 1.
    rgb_image : np.ndarray
        Input RGB image with shape ``(H, W, 3)``, values in ``[0, 1]``.
    patch_size : int
        Side length of each square patch in pixels.
    stride : int or None
        Step between consecutive patches. Defaults to
        ``patch_size // 2`` (50% overlap).

    Returns
    -------
    np.ndarray
        Blended ``(mu, log_var)`` prediction, shape ``(H, W, 2)``.

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

    prediction = np.zeros((ph, pw, 2), dtype=np.float32)
    weight_map = np.zeros((ph, pw, 1), dtype=np.float32)
    window = _gaussian_window(patch_size)[..., np.newaxis]  # (P, P, 1)

    for y in _patch_starts(ph, patch_size, stride):
        for x in _patch_starts(pw, patch_size, stride):
            patch = rgb_padded[y : y + patch_size, x : x + patch_size]
            patch_t = tf.expand_dims(patch, axis=0)  # (1, P, P, 3)
            pred = model(patch_t, training=False).numpy()[0]  # (P, P, 2)
            prediction[y : y + patch_size, x : x + patch_size] += pred * window
            weight_map[y : y + patch_size, x : x + patch_size] += window

    blended = prediction / (weight_map + 1e-8)
    return blended[:h, :w]
