"""Unit tests for scripts.delta_analysis."""

import numpy as np
import pytest

from scripts.delta_analysis import (
    analyze_delta,
    compute_local_stats,
    compute_ssim_components,
)


def _textured_image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, size), np.linspace(0, 4 * np.pi, size))
    texture = 0.5 + 0.3 * np.sin(x) * np.cos(y)
    noise = rng.normal(0, 0.02, size=(size, size))
    return np.clip(texture + noise, 0.0, 1.0).astype(np.float32)


def test_structure_map_is_one_for_identical_images() -> None:
    # Arrange
    real = _textured_image(0)

    # Act
    stats = compute_local_stats(real, real)
    components = compute_ssim_components(stats)

    # Assert
    assert components.structure == pytest.approx(1.0, abs=1e-4)


def test_structural_delta_is_low_for_constant_offset() -> None:
    # A uniform gray-level shift (substrate/acquisition effect) preserves
    # local structure, so the structural delta should stay near zero even
    # though the raw delta is large everywhere.
    # Arrange
    real = _textured_image(0)
    pred = np.clip(real - 0.2, 0.0, 1.0)

    # Act
    stats = compute_local_stats(real, pred)
    components = compute_ssim_components(stats)
    structural_delta = 1.0 - components.structure
    raw_delta = np.abs(real - pred)

    # Assert
    assert float(np.mean(structural_delta)) < 0.05
    assert float(np.mean(raw_delta)) > 0.15


def test_structural_delta_is_high_where_structure_is_replaced() -> None:
    # A zone overwritten with an uncorrelated pattern (a genuine hidden
    # mark) should raise the structural delta specifically in that zone.
    # Arrange
    real = _textured_image(0)
    pred = real.copy()
    rng = np.random.default_rng(1)
    pred[20:40, 20:40] = rng.random((20, 20)).astype(np.float32)

    # Act
    stats = compute_local_stats(real, pred)
    components = compute_ssim_components(stats)
    structural_delta = 1.0 - components.structure

    # Assert
    altered_zone = structural_delta[25:35, 25:35]
    untouched_zone = structural_delta[0:10, 0:10]
    assert float(np.mean(altered_zone)) > float(np.mean(untouched_zone))


def test_analyze_delta_returns_consistent_shapes() -> None:
    # Arrange
    real = _textured_image(0)
    pred = _textured_image(1)

    # Act
    result = analyze_delta(real, pred)

    # Assert
    for field in (
        result.raw_delta,
        result.luminance_map,
        result.contrast_map,
        result.structure_map,
        result.structural_delta,
        result.confidence_map,
    ):
        assert field.shape == real.shape


def test_analyze_delta_accepts_channel_dimension() -> None:
    # Arrange
    real = _textured_image(0)[..., np.newaxis]
    pred = _textured_image(1)[..., np.newaxis]

    # Act
    result = analyze_delta(real, pred)

    # Assert
    assert result.raw_delta.shape == real.shape[:2]


def test_confidence_map_is_high_for_identical_images() -> None:
    # Arrange
    real = _textured_image(0)

    # Act
    result = analyze_delta(real, real)

    # Assert
    assert float(np.mean(result.confidence_map)) > 0.9
