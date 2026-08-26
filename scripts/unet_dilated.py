"""UNet with a dilated (multi-scale) bottleneck — Round 3 exploratory variant.

Identical to :func:`scripts.unet.build_unet` except the bottleneck block:
see ``scripts.aspp`` for why (``fixing.md`` #7). Saved as a distinct
``unet_dilated`` checkpoint rather than overwriting ``unet``, so the two
can be compared directly under the same Round 1 recipe otherwise.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.aspp import dilated_bottleneck
from scripts.config import settings
from scripts.unet import _conv_block


def build_unet_dilated(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    dilation_rates: tuple[int, ...] | None = None,
) -> tf.keras.Model:
    """Build a 4-level UNet whose bottleneck is a dilated ASPP block.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level. Defaults to
        ``[64, 128, 256, 512]`` — same as :func:`scripts.unet.build_unet`.
    bottleneck : int
        Number of output channels of the dilated bottleneck block.
    dilation_rates : tuple[int, ...] or None
        Dilation rate for each parallel bottleneck branch. Defaults to
        ``settings.DILATION_RATES``.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output shape
        ``(None, None, None, 1)`` with sigmoid activation.
    """
    if filters is None:
        filters = [64, 128, 256, 512]
    if dilation_rates is None:
        dilation_rates = settings.DILATION_RATES

    inputs = layers.Input(shape=(None, None, 3))

    skips: list[tf.Tensor] = []
    x = inputs
    for f in filters:
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = dilated_bottleneck(x, bottleneck, dilation_rates=tuple(dilation_rates))

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="unet_dilated")
