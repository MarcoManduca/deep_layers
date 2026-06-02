"""ResUNet: UNet with residual encoder / decoder blocks."""

import tensorflow as tf
from tensorflow.keras import layers


def _residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Residual block: Conv → BN → ReLU → Conv → BN → Add(shortcut) → ReLU.

    A 1×1 projection shortcut is always applied to match channel counts.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of output filters.

    Returns
    -------
    tf.Tensor
        Output feature map with shape ``(..., H, W, filters)``.
    """
    shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resunet(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
) -> tf.keras.Model:
    """Build a 4-level ResUNet for single-channel IR prediction.

    Residual blocks replace the standard double-conv blocks of UNet,
    improving gradient flow and stability on small datasets.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block.

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
        x = _residual_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = _residual_block(x, bottleneck)

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _residual_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="resunet")
