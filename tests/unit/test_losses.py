"""Unit tests for scripts.losses."""

import numpy as np
import pytest
import tensorflow as tf

from scripts.losses import (
    combined_loss,
    combined_loss_advanced,
    fft_loss,
    laplacian_pyramid_loss,
)


def _identical_pair() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    y = tf.constant(rng.random((1, 32, 32, 1)), dtype=tf.float32)
    return y, y


def _different_pair() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    yt = tf.constant(rng.random((1, 32, 32, 1)), dtype=tf.float32)
    yp = tf.constant(rng.random((1, 32, 32, 1)), dtype=tf.float32)
    return yt, yp


def test_combined_loss_is_zero_for_identical_images() -> None:
    # SSIM of identical images is 1 up to float32 precision, so the loss
    # collapses to ~0 rather than exactly 0.
    yt, yp = _identical_pair()
    assert float(combined_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(combined_loss()(yt, yp)) > 0.0


def test_laplacian_pyramid_loss_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(laplacian_pyramid_loss()(yt, yp)) == 0.0


def test_laplacian_pyramid_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(laplacian_pyramid_loss()(yt, yp)) > 0.0


def test_fft_loss_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(fft_loss()(yt, yp)) == 0.0


def test_fft_loss_is_non_negative_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(fft_loss()(yt, yp)) > 0.0


def test_combined_loss_advanced_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(combined_loss_advanced()(yt, yp)) == 0.0


def test_combined_loss_advanced_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(combined_loss_advanced()(yt, yp)) > 0.0


def test_loss_factories_set_keras_friendly_name() -> None:
    assert combined_loss().__name__ == "combined_loss"
    assert combined_loss_advanced().__name__ == "combined_loss_advanced"
    assert fft_loss().__name__ == "fft_loss"
    assert laplacian_pyramid_loss().__name__ == "laplacian_pyramid_loss"
