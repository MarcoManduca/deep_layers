"""Unit tests for scripts.unet_residual and scripts.residual_head."""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from scripts.residual_head import ClipToUnitStraightThrough, RGBToGray
from scripts.trainer import compile_model
from scripts.unet import build_unet
from scripts.unet_residual import build_unet_residual

_TINY = {"filters": [8, 16], "bottleneck": 32}


def test_rgb_to_gray_averages_the_three_channels() -> None:
    layer = RGBToGray()
    x = np.array([[[[0.0, 0.5, 1.0]]]], dtype="float32")

    y = layer(x).numpy()

    assert y.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(y, [[[[0.5]]]], atol=1e-6)


def test_clip_to_unit_clips_the_forward_value() -> None:
    layer = ClipToUnitStraightThrough()
    x = tf.constant([[-0.4, 0.0, 0.3, 1.0, 1.7]])

    y = layer(x).numpy()

    np.testing.assert_allclose(y, [[0.0, 0.0, 0.3, 1.0, 1.0]], atol=1e-6)


def test_clip_to_unit_passes_the_gradient_through_out_of_range() -> None:
    layer = ClipToUnitStraightThrough()
    x = tf.Variable([[-0.4, 0.3, 1.7]])

    with tf.GradientTape() as tape:
        y = tf.reduce_sum(layer(x))
    grad = tape.gradient(y, x).numpy()

    # A plain tf.clip_by_value would give [0.0, 1.0, 0.0] here — the two
    # out-of-range pixels would be permanently dead.
    np.testing.assert_allclose(grad, [[1.0, 1.0, 1.0]], atol=1e-6)


def test_unet_residual_output_shape_and_range() -> None:
    model = build_unet_residual(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_unet_residual_head_is_tanh_and_carries_no_sigmoid() -> None:
    model = build_unet_residual(**_TINY)

    residual = model.get_layer("residual")
    assert residual.activation is tf.keras.activations.tanh
    activations = [
        layer.activation for layer in model.layers if hasattr(layer, "activation")
    ]
    assert tf.keras.activations.sigmoid not in activations


def test_unet_residual_backbone_matches_unet_layer_for_layer() -> None:
    residual_model = build_unet_residual(**_TINY)
    baseline = build_unet(**_TINY)

    # Only the head differs: unet ends with one sigmoid Conv2D, this ends
    # with a tanh Conv2D plus RGBToGray + Add + clip.
    def backbone(model: tf.keras.Model) -> list[str]:
        return [
            type(layer).__name__
            for layer in model.layers
            if not isinstance(layer, RGBToGray | ClipToUnitStraightThrough | layers.Add)
        ][:-1]

    assert backbone(residual_model) == backbone(baseline)


def test_unet_residual_output_equals_gray_when_the_residual_is_zeroed() -> None:
    model = build_unet_residual(**_TINY)
    head = model.get_layer("residual")
    head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    np.testing.assert_allclose(y, x.mean(axis=-1, keepdims=True), atol=1e-6)


def test_unet_residual_accepts_arbitrary_input_sizes() -> None:
    model = build_unet_residual(**_TINY)

    small = model(np.random.rand(1, 16, 16, 3).astype("float32")).numpy()
    large = model(np.random.rand(1, 32, 48, 3).astype("float32")).numpy()

    assert small.shape == (1, 16, 16, 1)
    assert large.shape == (1, 32, 48, 1)


def test_unet_residual_survives_save_load_round_trip() -> None:
    model = build_unet_residual(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y_before = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.keras"
        model.save(path)
        reloaded = tf.keras.models.load_model(path)

    y_after = reloaded(x, training=False).numpy()
    np.testing.assert_allclose(y_before, y_after, atol=1e-6)


def test_unet_residual_takes_a_gradient_step_through_combined_loss() -> None:
    # 176px is the smallest side tf.image.ssim_multiscale accepts with its
    # default 5 power factors (see tests/unit/test_losses.py); CROP_SIZE is
    # 256, so training resolution clears it comfortably.
    model = compile_model(build_unet_residual(**_TINY), "unet_residual")
    rng = np.random.default_rng(0)
    x = rng.random((1, 176, 176, 3)).astype("float32")
    y = rng.random((1, 176, 176, 1)).astype("float32")
    before = model.get_layer("residual").get_weights()[0].copy()

    history = model.fit(x, y, epochs=1, verbose=0)

    after = model.get_layer("residual").get_weights()[0]
    assert np.isfinite(history.history["loss"][0])
    assert not np.allclose(before, after)
