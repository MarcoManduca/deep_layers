"""UNet v2: standard UNet with optional, independently-toggleable modifications.

Three changes are exposed as parameters so each can be ablated in isolation
against the ``unet.py`` baseline, rather than shipped as a single fused
variant:

- ``use_strided_conv``: replaces ``MaxPool2D`` with a learned stride-2
  convolution for encoder downsampling.
- ``use_upsample_conv``: replaces ``Conv2DTranspose`` with
  ``UpSampling2D`` (bilinear) + ``Conv2D`` for decoder upsampling, to avoid
  the checkerboard artifacts transposed convolution is prone to.
- ``dropout_rate``: adds ``SpatialDropout2D`` at the bottleneck and the
  first (deepest) decoder block only — not at every block. Applying it
  throughout would compound with the ``BatchNormalization`` used in every
  conv block: dropout noise present in training but absent at inference
  shifts the statistics BN relies on (Li et al., 2019,
  "Understanding the Disharmony between Dropout and Batch Normalization").
  Restricting it to the two deepest blocks keeps the regularization where
  overfitting risk is most concentrated (bottleneck) while leaving the
  shallow encoder levels — where fine underdrawing-relevant detail lives —
  untouched.

All three default to off/``0.0``, so ``build_unet_v2()`` with no arguments
is architecturally identical to :func:`scripts.unet.build_unet`.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class _Upsample2x(layers.Layer):
    """Bilinear-upsample the spatial dimensions of ``x`` by a factor of 2.

    ``layers.UpSampling2D`` inspects the *static* input shape at trace
    time and raises on a fully dynamic ``(None, None, C)`` input — the
    shape used throughout this model to stay fully convolutional at any
    resolution. This layer instead resizes from the *runtime* shape via
    ``tf.image.resize``, the same pattern ``_ResizeToMatch`` in
    ``scripts/efficientnet_unet.py`` uses for the same reason.

    The ``register_keras_serializable`` decorator is required so that
    ``tf.keras.models.load_model`` can reconstruct this layer from a
    saved ``.keras`` checkpoint without needing ``custom_objects``.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        target = tf.shape(inputs)[1:3] * 2
        return tf.image.resize(inputs, target, method="bilinear")


def _conv_block(x: tf.Tensor, filters: int, dropout_rate: float = 0.0) -> tf.Tensor:
    """Two consecutive Conv → BN → ReLU operations, with optional dropout.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of convolutional filters.
    dropout_rate : float
        If > 0, applies ``SpatialDropout2D(dropout_rate)`` after the block.

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
    if dropout_rate > 0.0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def _downsample(x: tf.Tensor, filters: int, use_strided_conv: bool) -> tf.Tensor:
    """Halve H/W via a learned stride-2 conv or plain max pooling."""
    if use_strided_conv:
        x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    return layers.MaxPool2D(2)(x)


def _upsample(x: tf.Tensor, filters: int, use_upsample_conv: bool) -> tf.Tensor:
    """Double H/W via bilinear upsample + conv, or transposed convolution."""
    if use_upsample_conv:
        x = _Upsample2x()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    return layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)


def build_unet_v2(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    use_strided_conv: bool = False,
    use_upsample_conv: bool = False,
    dropout_rate: float = 0.0,
) -> tf.keras.Model:
    """Build a 4-level UNet v2 for single-channel IR prediction.

    Same topology as :func:`scripts.unet.build_unet` (fully convolutional,
    input H/W must be divisible by ``2 ** len(filters)``), with three
    independently-toggleable modifications — see module docstring.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block.
    use_strided_conv : bool
        If ``True``, encoder downsampling uses a learned stride-2
        convolution instead of ``MaxPool2D``.
    use_upsample_conv : bool
        If ``True``, decoder upsampling uses bilinear ``UpSampling2D`` +
        ``Conv2D`` instead of ``Conv2DTranspose``.
    dropout_rate : float
        ``SpatialDropout2D`` rate applied at the bottleneck and the first
        decoder block only. ``0.0`` (default) disables it.

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
        x = _downsample(x, f, use_strided_conv)

    x = _conv_block(x, bottleneck, dropout_rate=dropout_rate)

    for i, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        x = _upsample(x, f, use_upsample_conv)
        x = layers.Concatenate()([x, skip])
        block_dropout = dropout_rate if i == 0 else 0.0
        x = _conv_block(x, f, dropout_rate=block_dropout)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="unet_v2")
