"""UNet v2 with a dilated (multi-scale) bottleneck — Round 3 exploratory variant.

Identical to :func:`scripts.unet_v2.build_unet_v2` except the bottleneck
block: see ``scripts.aspp`` for why (``fixing.md`` #7). Saved as a distinct
``unet_v2_dilated`` checkpoint rather than overwriting ``unet_v2``. The
``use_strided_conv``/``use_upsample_conv``/``dropout_rate`` toggles are
carried over unchanged (same meaning as in ``unet_v2.py``) so this variant
can be trained with the exact same non-bottleneck configuration as
whichever ``unet_v2`` run it's compared against — only the bottleneck
differs between the two.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.aspp import dilated_bottleneck
from scripts.config import settings
from scripts.unet_v2 import _conv_block, _downsample, _upsample


def build_unet_v2_dilated(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    use_strided_conv: bool = False,
    use_upsample_conv: bool = False,
    dropout_rate: float = 0.0,
    dilation_rates: tuple[int, ...] | None = None,
) -> tf.keras.Model:
    """Build a 4-level UNet v2 whose bottleneck is a dilated ASPP block.

    Same topology and toggles as :func:`scripts.unet_v2.build_unet_v2`;
    only the bottleneck block changes. See that function's docstring for
    ``use_strided_conv``/``use_upsample_conv``/``dropout_rate``.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level. Defaults to
        ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of output channels of the dilated bottleneck block.
    use_strided_conv : bool
        If ``True``, encoder downsampling uses a learned stride-2
        convolution instead of ``MaxPool2D``.
    use_upsample_conv : bool
        If ``True``, decoder upsampling uses bilinear ``UpSampling2D`` +
        ``Conv2D`` instead of ``Conv2DTranspose``.
    dropout_rate : float
        ``SpatialDropout2D`` rate applied after the dilated bottleneck and
        the first decoder block only, same placement as ``unet_v2``.
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
        x = _downsample(x, f, use_strided_conv)

    x = dilated_bottleneck(x, bottleneck, dilation_rates=tuple(dilation_rates))
    if dropout_rate > 0.0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    for i, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        x = _upsample(x, f, use_upsample_conv)
        x = layers.Concatenate()([x, skip])
        block_dropout = dropout_rate if i == 0 else 0.0
        x = _conv_block(x, f, dropout_rate=block_dropout)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="unet_v2_dilated")
