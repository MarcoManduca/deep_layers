"""EfficientNetB0-UNet with a heteroscedastic (mu, log-variance) output head.

New architecture, kept separate from ``efficientnet_unet.py``: it reuses
the same pretrained EfficientNetB0 encoder, skip connections, and resize
helper (imported directly rather than duplicated — ``_ResizeToMatch`` is
``register_keras_serializable``-decorated, and registering a second class
under the same package/name would be ambiguous for
``tf.keras.models.load_model``), but the final ``1x1`` conv is duplicated
into two parallel branches instead of a single deterministic prediction —
a mean ``mu`` (same role as the existing model's output) and a
log-variance ``log_var`` (the model's own learned estimate of how
ambiguous the local RGB context has historically been). Trained with
:func:`scripts.losses.gaussian_nll_loss` via ``scripts/trainer_nll.py``.
See ``code-review.md`` §7.6 for the design rationale and references.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.config import settings
from scripts.efficientnet_unet import (
    _BOTTLENECK_LAYER,
    _SKIP_LAYER_NAMES,
    _conv_block,
    _ResizeToMatch,
)
from scripts.nll_layers import ClipLogVar


def build_efficientnet_unet_nll(
    decoder_filters: list[int] | None = None,
    freeze_encoder: bool = True,
    log_var_min: float | None = None,
    log_var_max: float | None = None,
) -> tf.keras.Model:
    """Build a UNet with a pretrained EfficientNetB0 encoder and an NLL head.

    Predicts per-pixel ``(mu, log_var)``. Identical encoder/decoder structure to
    :func:`scripts.efficientnet_unet.build_efficientnet_unet`; only the
    output head differs. ``mu`` is the expected IR value (same
    range/activation as the deterministic model); ``log_var`` is the
    model's own learned estimate of aleatoric uncertainty for that
    prediction, clipped to ``[log_var_min, log_var_max]`` for numerical
    stability before being consumed by
    :func:`scripts.losses.gaussian_nll_loss`.

    Input images must be normalised to ``[0, 1]``; the model applies an
    internal ``× 255`` rescaling before the EfficientNet stem, matching
    the range expected by ImageNet pre-trained weights.

    Parameters
    ----------
    decoder_filters : list[int] or None
        Five channel counts for the decoder path. See
        :func:`scripts.efficientnet_unet.build_efficientnet_unet` for the
        per-index meaning. Defaults to ``[256, 128, 64, 32, 16]``.
    freeze_encoder : bool
        If ``True`` (default), EfficientNetB0 weights are frozen.
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
    if decoder_filters is None:
        decoder_filters = [256, 128, 64, 32, 16]
    if log_var_min is None:
        log_var_min = settings.NLL_LOG_VAR_MIN
    if log_var_max is None:
        log_var_max = settings.NLL_LOG_VAR_MAX

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )
    backbone.trainable = not freeze_encoder

    skip_outputs = [backbone.get_layer(n).output for n in _SKIP_LAYER_NAMES]
    bottleneck_output = backbone.get_layer(_BOTTLENECK_LAYER).output
    encoder = tf.keras.Model(
        inputs=backbone.input,
        outputs=[*skip_outputs, bottleneck_output],
        name="efficientnet_b0_encoder_nll",
    )

    inputs = layers.Input(shape=(None, None, 3))
    # EfficientNetB0 expects [0, 255]; rescale from normalised [0, 1].
    encoder_outputs = encoder(inputs * 255.0)
    s1, s2, s3, s4 = (
        encoder_outputs[0],
        encoder_outputs[1],
        encoder_outputs[2],
        encoder_outputs[3],
    )
    x = encoder_outputs[4]  # H/32, 1280 ch

    # Bottleneck: reduce 1280 → decoder_filters[0] channels.
    x = _conv_block(x, decoder_filters[0])

    # Decoder: four upsampling stages paired with encoder skip connections.
    for f, skip in zip(decoder_filters[1:], [s4, s3, s2, s1]):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = _ResizeToMatch()([x, skip])
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    # Final upsample H/2 → H (no skip connection at full resolution).
    x = layers.Conv2DTranspose(decoder_filters[-1], 2, strides=2, padding="same")(x)
    x = _conv_block(x, decoder_filters[-1])

    mu = layers.Conv2D(1, 1, activation="sigmoid", name="mu")(x)
    log_var = layers.Conv2D(1, 1, activation="linear", name="log_var_raw")(x)
    log_var = ClipLogVar(log_var_min, log_var_max, name="log_var")(log_var)
    outputs = layers.Concatenate(name="mu_log_var")([mu, log_var])

    return tf.keras.Model(inputs, outputs, name="efficientnet_unet_nll")
