"""Unit tests for the independently-toggleable unet_v2 modifications."""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from scripts.unet_v2 import _Upsample2x, build_unet_v2

_TINY = {"filters": [8, 16], "bottleneck": 32}


def _layer_types(model) -> list[type]:
    return [type(layer) for layer in model.layers]


def test_default_flags_match_baseline_unet_topology() -> None:
    model = build_unet_v2(**_TINY)

    types = _layer_types(model)
    assert layers.MaxPool2D in types
    assert layers.Conv2DTranspose in types
    assert layers.SpatialDropout2D not in types


def test_strided_conv_replaces_max_pooling() -> None:
    model = build_unet_v2(**_TINY, use_strided_conv=True)

    types = _layer_types(model)
    assert layers.MaxPool2D not in types
    strided_convs = [
        layer
        for layer in model.layers
        if isinstance(layer, layers.Conv2D) and layer.strides == (2, 2)
    ]
    assert len(strided_convs) == len(_TINY["filters"])


def test_upsample_conv_replaces_transposed_conv() -> None:
    model = build_unet_v2(**_TINY, use_upsample_conv=True)

    types = _layer_types(model)
    assert layers.Conv2DTranspose not in types
    assert _Upsample2x in types


def test_dropout_applied_only_at_bottleneck_and_first_decoder_block() -> None:
    model = build_unet_v2(**_TINY, dropout_rate=0.2)

    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, layers.SpatialDropout2D)
    ]
    assert len(dropout_layers) == 2
    assert all(layer.rate == 0.2 for layer in dropout_layers)


def test_zero_dropout_rate_adds_no_dropout_layers() -> None:
    model = build_unet_v2(**_TINY, dropout_rate=0.0)

    types = _layer_types(model)
    assert layers.SpatialDropout2D not in types


def test_all_modifications_combined_output_shape_and_range() -> None:
    model = build_unet_v2(
        **_TINY,
        use_strided_conv=True,
        use_upsample_conv=True,
        dropout_rate=0.1,
    )
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_upsample_conv_variant_survives_save_load_round_trip() -> None:
    # _Upsample2x must be register_keras_serializable-decorated for
    # tf.keras.models.load_model to reconstruct it without custom_objects
    # (same bug class ClipLogVar/_ResizeToMatch were added to fix elsewhere).
    model = build_unet_v2(**_TINY, use_upsample_conv=True)
    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y_before = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)

    y_after = reloaded(x, training=False).numpy()
    np.testing.assert_allclose(y_before, y_after, atol=1e-6)
