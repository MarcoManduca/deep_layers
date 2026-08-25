"""Unit tests for scripts.losses."""

import numpy as np
import pytest
import tensorflow as tf

from scripts.losses import (
    charbonnier_loss,
    combined_loss,
    combined_loss_advanced,
    fft_loss,
    laplacian_pyramid_loss,
    ms_ssim_loss,
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


# tf.image.ssim_multiscale needs >= ~176px per side for its default 5
# scales (11x11 filter) — smaller inputs raise inside TF. Real training
# images are always far larger after padding, so this only affects tests.
def _identical_pair_ms_ssim() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    y = tf.constant(rng.random((1, 176, 176, 1)), dtype=tf.float32)
    return y, y


def _different_pair_ms_ssim() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    yt = tf.constant(rng.random((1, 176, 176, 1)), dtype=tf.float32)
    yp = tf.constant(rng.random((1, 176, 176, 1)), dtype=tf.float32)
    return yt, yp


def test_charbonnier_loss_is_near_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(charbonnier_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_charbonnier_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(charbonnier_loss()(yt, yp)) > 0.0


def test_charbonnier_loss_sets_keras_friendly_name() -> None:
    assert charbonnier_loss().__name__ == "charbonnier_loss"


def test_ms_ssim_loss_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair_ms_ssim()
    assert float(ms_ssim_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_ms_ssim_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair_ms_ssim()
    assert float(ms_ssim_loss()(yt, yp)) > 0.0


def test_ms_ssim_loss_sets_keras_friendly_name() -> None:
    assert ms_ssim_loss().__name__ == "ms_ssim_loss"


def test_combined_loss_is_zero_for_identical_images() -> None:
    # MS-SSIM of identical images is 1 up to float32 precision, so the
    # loss collapses to ~0 rather than exactly 0.
    yt, yp = _identical_pair_ms_ssim()
    assert float(combined_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair_ms_ssim()
    assert float(combined_loss()(yt, yp)) > 0.0


def test_combined_loss_weights_ms_ssim_more_than_charbonnier() -> None:
    # fixing.md #9: alpha=0.16 weights the Charbonnier term, so the
    # default loss should track (1 - MS-SSIM) more closely than a
    # pure-Charbonnier loss would.
    yt, yp = _different_pair_ms_ssim()
    combined = float(combined_loss()(yt, yp))
    ms_ssim_only = float(ms_ssim_loss()(yt, yp))
    charbonnier_only = float(charbonnier_loss()(yt, yp))
    assert abs(combined - ms_ssim_only) < abs(combined - charbonnier_only)


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
