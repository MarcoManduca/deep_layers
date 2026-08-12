"""UNet with a heteroscedastic (mu, log-variance) output head.

New architecture, kept separate from ``unet.py``: it reuses the same
encoder/decoder structure, but the final ``1x1`` conv is duplicated into
two parallel branches instead of a single deterministic prediction — a
mean ``mu`` (same role as the existing model's output) and a log-variance
``log_var`` (the model's own learned estimate of how ambiguous the local
RGB context has historically been). Trained with
:func:`scripts.losses.gaussian_nll_loss` via ``scripts/trainer_nll.py``.
See ``code-review.md`` §7.6 for the design rationale and references.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.config import settings
from scripts.nll_layers import ClipLogVar


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two consecutive Conv → BN → ReLU operations.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of convolutional filters.

    Returns
    -------
    tf.Tensor
        Output feature map with shape ``(..., H, W, filters)``.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet_nll(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    log_var_min: float | None = None,
    log_var_max: float | None = None,
) -> tf.keras.Model:
    """Build a 4-level UNet predicting per-pixel ``(mu, log_var)``.

    Identical encoder/decoder structure to
    :func:`scripts.unet.build_unet`; only the output head differs. ``mu``
    is the expected IR value (same range/activation as the deterministic
    model); ``log_var`` is the model's own learned estimate of aleatoric
    uncertainty for that prediction, clipped to
    ``[log_var_min, log_var_max]`` for numerical stability before being
    consumed by :func:`scripts.losses.gaussian_nll_loss`.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block.
        Defaults to 1024.
    log_var_min : float or None
        Lower clip bound for the ``log_var`` channel. Defaults to
        ``settings.NLL_LOG_VAR_MIN``.
    log_var_max : float or None
        Upper clip bound for the ``log_var`` channel. Defaults to
        ``settings.NLL_LOG_VAR_MAX``.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output shape
        ``(None, None, None, 2)``: channel 0 is ``mu`` (sigmoid, values in
        ``[0, 1]``), channel 1 is ``log_var`` (linear, clipped).
    """
    if filters is None:
        filters = [64, 128, 256, 512]
    if log_var_min is None:
        log_var_min = settings.NLL_LOG_VAR_MIN
    if log_var_max is None:
        log_var_max = settings.NLL_LOG_VAR_MAX

    inputs = layers.Input(shape=(None, None, 3))

    skips: list[tf.Tensor] = []
    x = inputs
    for f in filters:
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = _conv_block(x, bottleneck)

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    mu = layers.Conv2D(1, 1, activation="sigmoid", name="mu")(x)
    log_var = layers.Conv2D(1, 1, activation="linear", name="log_var_raw")(x)
    log_var = ClipLogVar(log_var_min, log_var_max, name="log_var")(log_var)
    outputs = layers.Concatenate(name="mu_log_var")([mu, log_var])

    return tf.keras.Model(inputs, outputs, name="unet_nll")
