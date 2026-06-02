"""Custom loss functions for IR image prediction."""

from collections.abc import Callable

import tensorflow as tf


def combined_loss(alpha: float = 0.7) -> Callable:
    """Return a combined MAE + (1 - SSIM) loss function.

    The loss is defined as::

        loss = alpha * MAE(y_true, y_pred)
             + (1 - alpha) * (1 - SSIM(y_true, y_pred))

    MAE encourages pixel-level fidelity; the SSIM term preserves
    perceptual structure (edges, texture contrast).

    Parameters
    ----------
    alpha : float
        Weight for the MAE term. ``(1 - alpha)`` weights SSIM.
        Must be in ``[0, 1]``.

    Returns
    -------
    Callable
        Loss function with signature ``(y_true, y_pred) -> tf.Tensor``
        compatible with ``model.compile(loss=...)``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return alpha * mae + (1.0 - alpha) * (1.0 - ssim)

    loss.__name__ = "combined_loss"
    return loss
