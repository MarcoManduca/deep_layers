"""Attention UNet with a heteroscedastic (mu, log-variance) output head.

New architecture, kept separate from ``attention_unet.py``: it reuses the
same encoder / decoder / attention-gate structure, but the final ``1x1``
conv is duplicated into two parallel branches instead of a single
deterministic prediction — a mean ``mu`` (same role as the existing
model's output) and a log-variance ``log_var`` (the model's own learned
estimate of how ambiguous the local RGB context has historically been).
Trained with :func:`scripts.losses.gaussian_nll_loss` via
``scripts/trainer_nll.py``. See ``code-review.md`` §7.6 for the design
rationale and references.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.config import settings


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class ClipLogVar(layers.Layer):
    """Clip a log-variance tensor to ``[min_value, max_value]``.

    A small named, serializable layer instead of a ``Lambda`` layer:
    Keras refuses to deserialize ``Lambda`` layers wrapping a Python
    function by default (arbitrary-code-execution risk), which would
    otherwise break ``tf.keras.models.load_model`` on any saved
    checkpoint.

    Parameters
    ----------
    min_value : float
        Lower clip bound.
    max_value : float
        Upper clip bound.
    """

    def __init__(self, min_value: float, max_value: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


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


def _attention_gate(
    x: tf.Tensor,
    g: tf.Tensor,
    filters: int,
) -> tf.Tensor:
    """Additive soft-attention gate (Oktay et al., 2018).

    Computes a spatial attention map from the skip connection ``x``
    and the gating signal ``g`` (decoder feature map), then scales
    ``x`` element-wise.

    Parameters
    ----------
    x : tf.Tensor
        Skip connection from the encoder, shape ``(B, H, W, C_x)``.
    g : tf.Tensor
        Gating signal from the decoder, shape ``(B, H, W, C_g)``.
    filters : int
        Number of intermediate attention filters.

    Returns
    -------
    tf.Tensor
        Attended skip connection with the same shape as ``x``.
    """
    theta_x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    phi_g = layers.Conv2D(filters, 1, padding="same", use_bias=False)(g)

    add = layers.Add()([theta_x, phi_g])
    add = layers.ReLU()(add)

    psi = layers.Conv2D(1, 1, padding="same", use_bias=False)(add)
    psi = layers.Activation("sigmoid")(psi)

    return layers.Multiply()([x, psi])


def build_attention_unet_nll(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    log_var_min: float | None = None,
    log_var_max: float | None = None,
) -> tf.keras.Model:
    """Build a 4-level Attention UNet predicting per-pixel ``(mu, log_var)``.

    Identical encoder/decoder/attention-gate structure to
    :func:`scripts.attention_unet.build_attention_unet`; only the output
    head differs. ``mu`` is the expected IR value (same range/activation as
    the deterministic model); ``log_var`` is the model's own learned
    estimate of aleatoric uncertainty for that prediction, clipped to
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
        attended = _attention_gate(skip, x, filters=f // 2)
        x = layers.Concatenate()([x, attended])
        x = _conv_block(x, f)

    mu = layers.Conv2D(1, 1, activation="sigmoid", name="mu")(x)
    log_var = layers.Conv2D(1, 1, activation="linear", name="log_var_raw")(x)
    log_var = ClipLogVar(log_var_min, log_var_max, name="log_var")(log_var)
    outputs = layers.Concatenate(name="mu_log_var")([mu, log_var])

    return tf.keras.Model(inputs, outputs, name="attention_unet_nll")
