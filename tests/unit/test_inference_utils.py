"""Unit tests for scripts.inference_utils."""

import numpy as np
import pytest
import tensorflow as tf

from scripts.inference_utils import (
    _gaussian_window,
    _patch_starts,
    predict_with_overlap,
)


def test_gaussian_window_peaks_at_one() -> None:
    window = _gaussian_window(32)
    assert window.max() == pytest.approx(1.0)


def test_gaussian_window_has_requested_shape() -> None:
    window = _gaussian_window(24)
    assert window.shape == (24, 24)


def test_gaussian_window_is_symmetric() -> None:
    window = _gaussian_window(16)
    assert np.allclose(window, window.T)


@pytest.mark.parametrize(
    "size, patch_size, stride",
    [
        (100, 32, 16),
        (64, 64, 32),
        (50, 16, 16),
        (45, 32, 8),
    ],
)
def test_patch_starts_cover_every_pixel(
    size: int, patch_size: int, stride: int
) -> None:
    # Act
    starts = _patch_starts(size, patch_size, stride)
    covered = np.zeros(size, dtype=bool)
    for s in starts:
        covered[s : s + patch_size] = True

    # Assert
    assert covered.all()
    assert starts[-1] + patch_size >= size


def test_predict_with_overlap_rejects_patch_size_not_multiple_of_16(
    dummy_model: tf.keras.Model,
) -> None:
    image = np.random.rand(48, 48, 3).astype("float32")
    with pytest.raises(ValueError, match="multiple of 16"):
        predict_with_overlap(dummy_model, image, patch_size=30)


def test_predict_with_overlap_rejects_stride_larger_than_patch(
    dummy_model: tf.keras.Model,
) -> None:
    image = np.random.rand(48, 48, 3).astype("float32")
    with pytest.raises(ValueError, match="must be ≤ patch_size"):
        predict_with_overlap(dummy_model, image, patch_size=16, stride=32)


def test_predict_with_overlap_returns_input_resolution(
    dummy_model: tf.keras.Model,
) -> None:
    # Arrange
    image = np.random.rand(40, 50, 3).astype("float32")

    # Act
    pred = predict_with_overlap(dummy_model, image, patch_size=16, stride=8)

    # Assert
    assert pred.shape == (40, 50, 1)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0
