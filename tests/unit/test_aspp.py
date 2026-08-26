"""Unit tests for scripts.aspp's dilated bottleneck block."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from scripts.aspp import dilated_bottleneck


def test_output_shape_matches_requested_filters() -> None:
    x = layers.Input(shape=(16, 16, 8))
    y = dilated_bottleneck(x, filters=32, dilation_rates=(1, 2, 4))
    model = tf.keras.Model(x, y)

    out = model(np.random.rand(1, 16, 16, 8).astype("float32"), training=False)

    assert out.shape == (1, 16, 16, 32)


def test_uses_a_dilated_conv_per_rate() -> None:
    x = layers.Input(shape=(16, 16, 8))
    y = dilated_bottleneck(x, filters=16, dilation_rates=(1, 2, 4, 8))
    model = tf.keras.Model(x, y)

    dilations = sorted(
        layer.dilation_rate[0]
        for layer in model.layers
        if isinstance(layer, layers.Conv2D) and layer.kernel_size == (3, 3)
    )
    assert dilations == [1, 2, 4, 8]


def test_single_rate_skips_concatenate() -> None:
    x = layers.Input(shape=(16, 16, 8))
    y = dilated_bottleneck(x, filters=16, dilation_rates=(2,))
    model = tf.keras.Model(x, y)

    assert not any(isinstance(layer, layers.Concatenate) for layer in model.layers)
