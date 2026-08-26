"""Unit tests for scripts.unet_dilated and scripts.unet_v2_dilated."""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from scripts.unet_dilated import build_unet_dilated
from scripts.unet_v2_dilated import build_unet_v2_dilated

_TINY = {"filters": [8, 16], "bottleneck": 32, "dilation_rates": (1, 2)}


def test_unet_dilated_output_shape_and_range() -> None:
    model = build_unet_dilated(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_unet_dilated_has_no_max_pool_at_bottleneck_but_keeps_it_in_encoder() -> None:
    model = build_unet_dilated(**_TINY)

    pools = [layer for layer in model.layers if isinstance(layer, layers.MaxPool2D)]
    assert len(pools) == len(_TINY["filters"])

    dilated_convs = [
        layer
        for layer in model.layers
        if isinstance(layer, layers.Conv2D) and layer.dilation_rate != (1, 1)
    ]
    assert len(dilated_convs) > 0


def test_unet_dilated_survives_save_load_round_trip() -> None:
    model = build_unet_dilated(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y_before = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)

    y_after = reloaded(x, training=False).numpy()
    np.testing.assert_allclose(y_before, y_after, atol=1e-6)


def test_unet_v2_dilated_output_shape_and_range() -> None:
    model = build_unet_v2_dilated(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_unet_v2_dilated_default_kwargs_match_unet_v2_topology() -> None:
    model = build_unet_v2_dilated(
        **_TINY,
        use_strided_conv=True,
        use_upsample_conv=True,
        dropout_rate=0.2,
    )

    types = [type(layer) for layer in model.layers]
    assert layers.MaxPool2D not in types
    assert layers.Conv2DTranspose not in types
    assert layers.SpatialDropout2D in types


def test_unet_v2_dilated_survives_save_load_round_trip() -> None:
    model = build_unet_v2_dilated(**_TINY, use_upsample_conv=True)
    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y_before = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)

    y_after = reloaded(x, training=False).numpy()
    np.testing.assert_allclose(y_before, y_after, atol=1e-6)
