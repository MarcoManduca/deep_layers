"""Unit tests for scripts.stroke_stats."""

import numpy as np
import pytest

from scripts.stroke_stats import rank_by_coherence, stroke_coherence

_SIZE = 64


def _stroke() -> np.ndarray:
    """A single straight horizontal stroke — maximally oriented structure."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[30:34, :] = 1.0
    return signal


def _isotropic_noise() -> np.ndarray:
    return np.random.default_rng(0).random((_SIZE, _SIZE))


def test_a_straight_stroke_scores_high_coherence() -> None:
    assert stroke_coherence(_stroke()).coherence > 0.8


def test_isotropic_noise_scores_low_coherence() -> None:
    assert stroke_coherence(_isotropic_noise()).coherence < 0.3


def test_coherence_stays_in_the_unit_range() -> None:
    stats = stroke_coherence(_isotropic_noise())

    assert 0.0 <= stats.coherence <= 1.0
    assert stats.coherence_map.min() >= 0.0
    assert stats.coherence_map.max() <= 1.0


def test_coherence_is_invariant_to_signal_scale() -> None:
    signal = _stroke()

    # Both structure-tensor eigenvalues scale with the square of the signal,
    # so their normalised difference does not — which is what lets signals on
    # different scales (a raw delta vs. a z-score) be compared directly.
    assert stroke_coherence(signal * 10.0).coherence == pytest.approx(
        stroke_coherence(signal).coherence, rel=1e-6
    )


def test_gradient_energy_is_higher_for_a_stronger_signal() -> None:
    signal = _stroke()

    weak = stroke_coherence(signal).gradient_energy
    strong = stroke_coherence(signal * 10.0).gradient_energy

    assert strong > weak


def test_coherence_map_matches_the_input_shape() -> None:
    assert stroke_coherence(_stroke()).coherence_map.shape == (_SIZE, _SIZE)


def test_accepts_a_trailing_channel_axis() -> None:
    stats = stroke_coherence(_stroke()[..., np.newaxis])

    assert stats.coherence_map.shape == (_SIZE, _SIZE)


def test_larger_sigma_suppresses_fine_noise() -> None:
    noise = _isotropic_noise()

    assert stroke_coherence(noise, sigma=4.0).coherence < 0.3


def test_rank_by_coherence_puts_strokes_before_noise() -> None:
    signals = {"noise": _isotropic_noise(), "stroke": _stroke()}

    assert list(rank_by_coherence(signals)) == ["stroke", "noise"]


def test_rank_by_coherence_scores_every_candidate() -> None:
    signals = {"noise": _isotropic_noise(), "stroke": _stroke()}

    assert set(rank_by_coherence(signals)) == {"noise", "stroke"}
