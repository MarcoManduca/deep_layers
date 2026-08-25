"""Attention UNet: UNet with additive attention gates on skip connections."""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.norm_utils import num_groups as _num_groups


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two consecutive Conv → GroupNorm → ReLU operations.

    Uses ``GroupNormalization`` instead of ``BatchNormalization``
    (``fixing.md`` #1) and He init instead of Xavier (``fixing.md`` #3).

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
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
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


def build_attention_unet(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
) -> tf.keras.Model:
    """Build a 4-level Attention UNet for single-channel IR prediction.

    Attention gates on each skip connection suppress irrelevant encoder
    features, helping the model focus on underdrawing-relevant regions.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block.
        Defaults to 1024.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output
        shape ``(None, None, None, 1)`` with sigmoid activation.
    """
    if filters is None:
        filters = [64, 128, 256, 512]

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

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="attention_unet")
