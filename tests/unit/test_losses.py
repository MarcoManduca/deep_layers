"""Unit tests for scripts.losses."""

import numpy as np
import pytest
import tensorflow as tf

from scripts.delta_analysis import compute_local_stats
from scripts.losses import (
    _gaussian_blur,
    _gaussian_kernel_1d,
    combined_loss,
    combined_loss_advanced,
    combined_loss_normalized,
    fft_loss,
    laplacian_pyramid_loss,
    local_zscore_loss,
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
    assert combined_loss_normalized().__name__ == "combined_loss_normalized"
    assert local_zscore_loss().__name__ == "local_zscore_loss"
    assert fft_loss().__name__ == "fft_loss"
    assert laplacian_pyramid_loss().__name__ == "laplacian_pyramid_loss"


# ---------------------------------------------------------------------------
# Local windowed moments
# ---------------------------------------------------------------------------


def test_gaussian_kernel_is_normalised_and_symmetric() -> None:
    kernel = _gaussian_kernel_1d(11, 1.5)

    assert kernel.sum() == pytest.approx(1.0)
    assert kernel == pytest.approx(kernel[::-1])


def test_gaussian_blur_matches_numpy_local_mean() -> None:
    # Guards against drift between the TF windowing used by the loss and the
    # NumPy windowing used by scripts.delta_analysis: both must window the
    # image identically, or loss and post-hoc analysis stop being comparable.
    rng = np.random.default_rng(0)
    image = rng.random((24, 20)).astype(np.float32)

    kernel = tf.constant(_gaussian_kernel_1d(11, 1.5), dtype=tf.float32)
    tf_mean = _gaussian_blur(tf.constant(image)[tf.newaxis, ..., tf.newaxis], kernel, 1)
    numpy_mean = compute_local_stats(image, image, window_size=11, sigma=1.5).mu_real

    assert np.allclose(tf_mean[0, ..., 0].numpy(), numpy_mean, atol=1e-5)


def test_gaussian_blur_preserves_spatial_shape() -> None:
    x = tf.ones((1, 20, 24, 1))
    kernel = tf.constant(_gaussian_kernel_1d(11, 1.5), dtype=tf.float32)

    assert _gaussian_blur(x, kernel, 1).shape == (1, 20, 24, 1)


# ---------------------------------------------------------------------------
# Per-window z-score loss
# ---------------------------------------------------------------------------


def test_local_zscore_loss_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(local_zscore_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_local_zscore_loss_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(local_zscore_loss()(yt, yp)) > 0.0


def test_local_zscore_loss_is_invariant_to_local_affine_gray_shift() -> None:
    # The defining property: rescaling and offsetting the prediction leaves
    # the per-window z-score untouched, so the loss cannot see it.
    yt, _ = _identical_pair()
    yp = 1.2 * yt + 0.1

    assert float(local_zscore_loss()(yt, yp)) == pytest.approx(0.0, abs=1e-4)


def test_combined_loss_normalized_still_penalises_a_gray_shift() -> None:
    # Consequence of the invariance above: the MAE anchor is what keeps the
    # absolute gray level (and therefore the raw delta) meaningful.
    yt, _ = _identical_pair()
    yp = 1.2 * yt + 0.1

    assert float(combined_loss_normalized()(yt, yp)) > 0.05


def test_local_zscore_loss_is_zero_for_identical_flat_images() -> None:
    flat = tf.fill((1, 32, 32, 1), 0.4)
    assert float(local_zscore_loss()(flat, flat)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("noise_scale", [0.0, 1e-6])
def test_local_zscore_loss_gradient_is_finite_on_flat_images(
    noise_scale: float,
) -> None:
    # Without the variance floor the z-score divides by ~0 here and the sqrt
    # derivative is unbounded, which is the NaN case this loss must survive.
    rng = np.random.default_rng(0)
    yt = tf.fill((1, 32, 32, 1), 0.4)
    yp = tf.Variable(
        (0.3 + noise_scale * rng.random((1, 32, 32, 1))).astype(np.float32)
    )

    with tf.GradientTape() as tape:
        value = local_zscore_loss()(yt, yp)
    gradient = tape.gradient(value, yp)

    assert np.all(np.isfinite(gradient.numpy()))


def test_local_zscore_loss_clip_bounds_the_error_tail() -> None:
    yt, yp = _different_pair()

    unclipped = float(local_zscore_loss(clip_value=None)(yt, yp))
    clipped = float(local_zscore_loss(clip_value=0.5)(yt, yp))

    assert clipped < unclipped
    assert clipped <= 0.5


def test_combined_loss_normalized_is_zero_for_identical_images() -> None:
    yt, yp = _identical_pair()
    assert float(combined_loss_normalized()(yt, yp)) == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_normalized_is_positive_for_different_images() -> None:
    yt, yp = _different_pair()
    assert float(combined_loss_normalized()(yt, yp)) > 0.0


def test_combined_loss_normalized_reduces_to_mae_when_beta_is_zero() -> None:
    yt, yp = _different_pair()
    mae = float(tf.reduce_mean(tf.abs(yt - yp)))

    assert float(
        combined_loss_normalized(alpha=1.0, beta=0.0)(yt, yp)
    ) == pytest.approx(mae, abs=1e-6)
