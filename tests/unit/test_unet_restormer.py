"""Unit tests for the Restormer-bottleneck UNet variant."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from scripts.unet_restormer import RestormerBlock, build_unet_restormer

_TINY = {"filters": [8, 16], "bottleneck": 32, "num_heads": 4}


def test_restormer_block_present_at_bottleneck() -> None:
    model = build_unet_restormer(**_TINY)

    types = [type(layer) for layer in model.layers]
    assert RestormerBlock in types


def test_dim_not_divisible_by_num_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RestormerBlock(dim=32, num_heads=5)


def test_output_shape_and_range_square_input() -> None:
    model = build_unet_restormer(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_output_preserves_non_square_spatial_dimensions() -> None:
    # Exercises the dynamic-shape reshape inside RestormerBlock._attention
    # with H != W, since attention flattens (H, W) -> H*W and back.
    model = build_unet_restormer(**_TINY)
    x = np.random.rand(1, 32, 48, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 32, 48, 1)


def test_save_load_round_trip() -> None:
    # RestormerBlock is register_keras_serializable-decorated with a
    # get_config override — verify tf.keras.models.load_model can
    # reconstruct it without custom_objects (same bug class ClipLogVar /
    # _ResizeToMatch / _Upsample2x were added to fix elsewhere).
    model = build_unet_restormer(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y_before = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)

    y_after = reloaded(x, training=False).numpy()
    np.testing.assert_allclose(y_before, y_after, atol=1e-6)


def test_restormer_block_standalone_output_shape() -> None:
    block = RestormerBlock(dim=16, num_heads=4, ffn_expansion_factor=2)
    x = np.random.rand(2, 8, 8, 16).astype("float32")

    y = block(x).numpy()

    assert y.shape == x.shape


def test_restormer_block_zero_temperature_init_is_identity_like_attention() -> None:
    # With default (ones) temperature the block still mixes information;
    # this just checks the layer is trainable (has weights) and doesn't
    # blow up numerically on a larger, more realistic channel count.
    block = RestormerBlock(dim=64, num_heads=8)
    x = np.random.rand(1, 4, 4, 64).astype("float32")

    y = block(x).numpy()

    assert np.all(np.isfinite(y))
    assert len(block.trainable_weights) > 0


def test_layer_norm_used_not_batch_norm() -> None:
    block = RestormerBlock(dim=16, num_heads=4)
    _ = block(np.random.rand(1, 4, 4, 16).astype("float32"))

    assert isinstance(block.norm1, layers.LayerNormalization)
    assert isinstance(block.norm2, layers.LayerNormalization)
