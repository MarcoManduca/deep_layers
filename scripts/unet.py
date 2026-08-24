"""Standard UNet for RGB → IR image translation."""

import tensorflow as tf
from tensorflow.keras import layers


def _num_groups(filters: int, max_groups: int = 32) -> int:
    """Largest divisor of ``filters`` that is at most ``max_groups``.

    ``GroupNormalization`` requires ``groups`` to divide the channel count
    evenly; a fixed ``32`` (the paper's default) fails on the small filter
    counts unit tests use, so the group count adapts down.
    """
    groups = min(max_groups, filters)
    while filters % groups != 0:
        groups -= 1
    return groups


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two consecutive Conv → GroupNorm → ReLU operations.

    Uses ``GroupNormalization`` rather than ``BatchNormalization``: at
    ``settings.BATCH_SIZE == 8`` (well below the 50-256 range typical
    per-batch statistics are usually computed over), GroupNorm's
    per-example, per-channel-group statistics avoid the batch-size
    sensitivity BatchNorm has at this scale (see ``fixing.md`` #1).

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
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
    x = layers.ReLU()(x)
    return x


def build_unet(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
) -> tf.keras.Model:
    """Build a 4-level UNet for single-channel IR prediction.

    The model accepts images of any spatial size at both training and
    inference time (fully convolutional).  Input H and W must be
    divisible by ``2 ** len(filters)`` at training time; for arbitrary
    sizes use :func:`scripts.dataset.pad_to_multiple` beforehand.

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
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = _conv_block(x, bottleneck)

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="unet")
