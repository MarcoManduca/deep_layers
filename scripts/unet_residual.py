"""UNet with a residual output head: predicts IR as a delta from grayscale RGB.

Identical to :func:`scripts.unet.build_unet` — same encoder, bottleneck,
decoder, normalization and initialization, and it reuses that module's
``_conv_block`` directly — except for the **output head**:

- ``unet``: ``Conv2D(1, 1, activation="sigmoid")`` predicts the IR value
  directly, from scratch, at every pixel.
- ``unet_residual``: ``Conv2D(1, 1, activation="tanh")`` predicts a
  *residual* in ``[-1, 1]``, added to ``mean(R, G, B)`` and clipped back
  to ``[0, 1]``. The reference level is free (no parameters), so the
  network only has to learn where and how much IR departs from the
  visible-grey level.

The motivation is standard global residual learning (VDSR, Kim et al.
2016; DnCNN, Zhang et al. 2017): when input and target are largely
correlated, learning the difference is an easier optimization problem
than learning the target, and the high-frequency detail of the input is
preserved by construction rather than having to be reconstructed through
the bottleneck.

The bet is *not* free here, and this is why it is a separate checkpoint
rather than a change to ``unet``: ``mean(R, G, B)`` is a weak IR prior.
Infrared reflectance is not visible luminance — pigments with the same
grey value can be IR-opaque or IR-transparent (a dark blue is typically
transparent), so on this dataset the required residual may be large and
structured, which would erase the "the identity is free" advantage. That
is an empirical question about these paintings, and the point of training
the variant.

Saved under its own ``arch_name`` (``models/deterministic/unet_residual/``)
so the ``unet`` baseline is never overwritten and the two differ *only* in
the head — the same discipline the Round 3 dilated variants follow.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.residual_head import ClipToUnitStraightThrough, RGBToGray
from scripts.unet import _conv_block


def build_unet_residual(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
) -> tf.keras.Model:
    """Build a 4-level UNet that predicts IR as a residual from grayscale RGB.

    Fully convolutional, like every other architecture here: the input
    shape is ``(None, None, 3)`` and H/W must be divisible by
    ``2 ** len(filters)`` — use :func:`scripts.dataset.pad_to_multiple`
    for arbitrary sizes.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]`` — same as
        :func:`scripts.unet.build_unet`.
    bottleneck : int
        Number of filters in the bottleneck block.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output shape
        ``(None, None, None, 1)`` in ``[0, 1]``. The range is enforced by
        :class:`scripts.residual_head.ClipToUnitStraightThrough` rather
        than by a ``sigmoid``, so it is exact rather than asymptotic.
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
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    # tanh, not sigmoid: the residual must be signed — an IR pixel can be
    # darker (underdrawing showing through) or lighter than the visible
    # grey level. Glorot (the Keras default) rather than the he_normal
    # used inside the conv blocks, since he_normal assumes a ReLU-like
    # activation (fixing.md #3) and this head is tanh.
    residual = layers.Conv2D(1, 1, activation="tanh", name="residual")(x)

    gray = RGBToGray(name="rgb_gray")(inputs)
    outputs = ClipToUnitStraightThrough(name="ir")(layers.Add()([gray, residual]))

    return tf.keras.Model(inputs, outputs, name="unet_residual")
