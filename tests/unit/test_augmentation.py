"""Unit tests for scripts.augmentation."""

import numpy as np
import tensorflow as tf

from scripts.augmentation import augment_pair


def _make_pair() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    rgb = tf.constant(rng.random((16, 16, 3)), dtype=tf.float32)
    ir = tf.constant(rng.random((16, 16, 1)), dtype=tf.float32)
    return rgb, ir


def test_augment_pair_is_deterministic_for_same_seed() -> None:
    # Arrange
    rgb, ir = _make_pair()
    seed = tf.constant([42, 0])

    # Act
    rgb_a, ir_a = augment_pair(rgb, ir, seed=seed)
    rgb_b, ir_b = augment_pair(rgb, ir, seed=seed)

    # Assert
    assert np.allclose(rgb_a, rgb_b)
    assert np.allclose(ir_a, ir_b)


def test_augment_pair_differs_for_different_seed() -> None:
    # Arrange
    rgb, ir = _make_pair()

    # Act
    rgb_a, _ = augment_pair(rgb, ir, seed=tf.constant([42, 0]))
    rgb_b, _ = augment_pair(rgb, ir, seed=tf.constant([42, 1]))

    # Assert
    assert not np.allclose(rgb_a, rgb_b)


def test_augment_pair_preserves_shapes() -> None:
    # Arrange
    rgb, ir = _make_pair()

    # Act
    rgb_out, ir_out = augment_pair(rgb, ir, seed=tf.constant([1, 2]))

    # Assert
    assert rgb_out.shape == rgb.shape
    assert ir_out.shape == ir.shape


def test_augment_pair_keeps_rgb_in_unit_range() -> None:
    # Arrange
    rgb, ir = _make_pair()

    # Act
    rgb_out, _ = augment_pair(rgb, ir, seed=tf.constant([3, 4]))

    # Assert
    assert float(tf.reduce_min(rgb_out)) >= 0.0
    assert float(tf.reduce_max(rgb_out)) <= 1.0


def test_augment_pair_only_flips_ir_without_photometric_change() -> None:
    # Arrange — IR receives spatial flips only, so its pixel multiset is preserved.
    rgb, ir = _make_pair()

    # Act
    _, ir_out = augment_pair(rgb, ir, seed=tf.constant([5, 6]))

    # Assert
    assert np.allclose(np.sort(ir.numpy().ravel()), np.sort(ir_out.numpy().ravel()))


def test_augment_pair_crops_to_requested_size() -> None:
    # Arrange
    rgb, ir = _make_pair()

    # Act
    rgb_out, ir_out = augment_pair(rgb, ir, seed=tf.constant([7, 8]), crop_size=8)

    # Assert
    assert rgb_out.shape == (8, 8, 3)
    assert ir_out.shape == (8, 8, 1)


def test_augment_pair_crops_rgb_and_ir_with_the_same_box() -> None:
    # Arrange — a strictly increasing ramp (kept in [0.2, 0.6] so photometric
    # jitter never saturates the clip) gives a unique brightest pixel. RGB
    # photometric jitter is monotone, so it preserves the argmax position.
    # IR and RGB share the ramp; if crop or flips used different boxes the
    # brightest-pixel locations would diverge.
    ramp = np.linspace(0.2, 0.6, 16 * 16).reshape(16, 16).astype("float32")
    rgb = tf.constant(np.stack([ramp, ramp, ramp], axis=-1))
    ir = tf.constant(ramp[..., np.newaxis])

    # Act
    rgb_out, ir_out = augment_pair(rgb, ir, seed=tf.constant([9, 10]), crop_size=8)
    rgb_argmax = np.unravel_index(np.argmax(rgb_out.numpy()[..., 0]), (8, 8))
    ir_argmax = np.unravel_index(np.argmax(ir_out.numpy()[..., 0]), (8, 8))

    # Assert
    assert rgb_argmax == ir_argmax


def test_augment_pair_is_deterministic_with_crop() -> None:
    # Arrange
    rgb, ir = _make_pair()
    seed = tf.constant([11, 12])

    # Act
    a = augment_pair(rgb, ir, seed=seed, crop_size=8)[0]
    b = augment_pair(rgb, ir, seed=seed, crop_size=8)[0]

    # Assert
    assert np.allclose(a, b)
