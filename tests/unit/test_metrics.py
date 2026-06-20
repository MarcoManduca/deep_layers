"""Unit tests for scripts.metrics."""

import numpy as np
import pytest
import tensorflow as tf

from scripts.metrics import PSNRMetric, SSIMMetric


def _batch(seed: int) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    return tf.constant(rng.random((2, 16, 16, 1)), dtype=tf.float32)


def test_ssim_metric_is_one_for_identical_images() -> None:
    # Arrange
    metric = SSIMMetric()
    y = _batch(0)

    # Act
    metric.update_state(y, y)

    # Assert
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)


def test_ssim_metric_below_one_for_different_images() -> None:
    # Arrange
    metric = SSIMMetric()

    # Act
    metric.update_state(_batch(0), _batch(1))

    # Assert
    assert float(metric.result()) < 1.0


def test_psnr_metric_accumulates_mean_across_updates() -> None:
    # Arrange
    metric = PSNRMetric()
    yt, yp = _batch(0), _batch(1)

    # Act
    metric.update_state(yt, yp)
    single = float(metric.result())
    metric.update_state(yt, yp)
    doubled = float(metric.result())

    # Assert — identical batches keep the running mean unchanged.
    assert doubled == pytest.approx(single, rel=1e-5)


def test_metric_reset_state_clears_accumulators() -> None:
    # Arrange
    metric = SSIMMetric()
    metric.update_state(_batch(0), _batch(1))

    # Act
    metric.reset_state()
    metric.update_state(_batch(0), _batch(0))

    # Assert
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)
