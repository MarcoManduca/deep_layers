"""Unit tests for scripts.calibration."""

import math

import numpy as np
import pytest

from scripts.calibration import (
    coverage_probability,
    dispersion,
    error_sigma_correlation,
    evaluate_calibration,
    learned_zscore,
    mean_gaussian_nll,
    nominal_coverage,
    sharpness,
    sigma_reliability,
    structural_zscore,
)

_SIZE = (200, 200)


def _calibrated_pair(
    sigma_value: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``real = mu + sigma * N(0, 1)`` — a perfectly calibrated prediction."""
    rng = np.random.default_rng(seed)
    mu = np.full(_SIZE, 0.5, dtype=np.float32)
    sigma = np.full(_SIZE, sigma_value, dtype=np.float32)
    real = mu + sigma * rng.standard_normal(_SIZE).astype(np.float32)
    return real, mu, sigma


def _heteroscedastic_pair(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same, with ``sigma`` varying smoothly across the image."""
    rng = np.random.default_rng(seed)
    mu = np.full(_SIZE, 0.5, dtype=np.float32)
    sigma = np.linspace(0.02, 0.2, _SIZE[0] * _SIZE[1], dtype=np.float32).reshape(_SIZE)
    real = mu + sigma * rng.standard_normal(_SIZE).astype(np.float32)
    return real, mu, sigma


# --- z-score -----------------------------------------------------------------


def test_learned_zscore_matches_manual_computation() -> None:
    real = np.array([[0.5, 0.8], [0.4, 0.5]], dtype=np.float32)
    mu = np.array([[0.5, 0.6], [0.5, 0.5]], dtype=np.float32)
    sigma = np.array([[0.1, 0.1], [0.2, 0.1]], dtype=np.float32)

    z = learned_zscore(real, mu, sigma)

    np.testing.assert_allclose(z, [[0.0, 2.0], [-0.5, 0.0]], atol=1e-4)


def test_learned_zscore_drops_the_trailing_channel_axis() -> None:
    real, mu, sigma = _calibrated_pair()

    z = learned_zscore(real[..., None], mu[..., None], sigma[..., None])

    assert z.shape == _SIZE


# --- coverage ----------------------------------------------------------------


@pytest.mark.parametrize(
    "k, expected",
    [(1.0, 0.6827), (2.0, 0.9545), (3.0, 0.9973)],
)
def test_nominal_coverage_matches_the_gaussian_reference(
    k: float, expected: float
) -> None:
    assert nominal_coverage(k) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_coverage_probability_matches_nominal_for_calibrated_prediction(
    k: float,
) -> None:
    real, mu, sigma = _calibrated_pair()

    coverage = coverage_probability(real, mu, sigma, k=k)

    assert coverage.empirical == pytest.approx(coverage.nominal, abs=0.01)


def test_coverage_probability_is_below_nominal_for_overconfident_prediction() -> None:
    real, mu, sigma = _calibrated_pair()

    coverage = coverage_probability(real, mu, sigma / 4.0, k=1.0)

    assert coverage.error < -0.2


def test_coverage_probability_is_above_nominal_for_underconfident_prediction() -> None:
    real, mu, sigma = _calibrated_pair()

    coverage = coverage_probability(real, mu, sigma * 4.0, k=1.0)

    assert coverage.error > 0.2


# --- error/sigma correlation -------------------------------------------------


def test_error_sigma_correlation_is_one_when_sigma_orders_the_errors() -> None:
    mu = np.zeros(_SIZE, dtype=np.float32)
    error = np.linspace(0.01, 1.0, _SIZE[0] * _SIZE[1], dtype=np.float32).reshape(_SIZE)

    correlation = error_sigma_correlation(error, mu, error)

    assert correlation == pytest.approx(1.0, abs=1e-6)


def test_error_sigma_correlation_is_zero_for_constant_sigma() -> None:
    real, mu, sigma = _calibrated_pair()

    assert error_sigma_correlation(real, mu, sigma) == 0.0


def test_error_sigma_correlation_is_reproducible_when_subsampling() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    first = error_sigma_correlation(real, mu, sigma, max_samples=5_000, seed=7)
    second = error_sigma_correlation(real, mu, sigma, max_samples=5_000, seed=7)

    assert first == second


# --- reliability -------------------------------------------------------------


def test_sigma_reliability_lies_on_the_identity_for_calibrated_prediction() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    curve = sigma_reliability(real, mu, sigma, n_bins=10)

    assert curve.ence < 0.05
    np.testing.assert_allclose(curve.observed_error, curve.predicted_sigma, rtol=0.15)


def test_sigma_reliability_ence_is_large_for_overconfident_prediction() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    curve = sigma_reliability(real, mu, sigma / 4.0, n_bins=10)

    assert curve.ence > 1.0


def test_sigma_reliability_collapses_to_one_bin_for_constant_sigma() -> None:
    real, mu, sigma = _calibrated_pair()

    curve = sigma_reliability(real, mu, sigma, n_bins=10)

    assert curve.predicted_sigma.shape == (1,)
    assert curve.counts.sum() == real.size


def test_sigma_reliability_returns_the_requested_number_of_bins() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    curve = sigma_reliability(real, mu, sigma, n_bins=8)

    assert curve.predicted_sigma.shape == (8,)


# --- sharpness / dispersion --------------------------------------------------


def test_sharpness_returns_the_mean_sigma() -> None:
    _, _, sigma = _calibrated_pair(sigma_value=0.25)

    assert sharpness(sigma) == pytest.approx(0.25, abs=1e-6)


def test_dispersion_is_zero_for_a_constant_sigma() -> None:
    _, _, sigma = _calibrated_pair()

    assert dispersion(sigma) == pytest.approx(0.0, abs=1e-6)


def test_dispersion_is_positive_for_a_varying_sigma() -> None:
    _, _, sigma = _heteroscedastic_pair()

    assert dispersion(sigma) > 0.1


# --- likelihood --------------------------------------------------------------


def test_mean_gaussian_nll_matches_the_manual_formula() -> None:
    real = np.array([[0.7]], dtype=np.float32)
    mu = np.array([[0.5]], dtype=np.float32)
    sigma = np.array([[0.1]], dtype=np.float32)
    expected = 0.5 * math.log(2 * math.pi * 0.01) + (0.2**2) / (2 * 0.01)

    assert mean_gaussian_nll(real, mu, sigma) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("wrong_scale", [0.25, 4.0])
def test_mean_gaussian_nll_is_lowest_at_the_true_sigma(wrong_scale: float) -> None:
    real, mu, sigma = _calibrated_pair()

    calibrated = mean_gaussian_nll(real, mu, sigma)
    miscalibrated = mean_gaussian_nll(real, mu, sigma * wrong_scale)

    assert calibrated < miscalibrated


# --- aggregate ---------------------------------------------------------------


def test_evaluate_calibration_reports_unit_z_std_for_calibrated_prediction() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    result = evaluate_calibration(real, mu, sigma)

    assert result.z_mean == pytest.approx(0.0, abs=0.02)
    assert result.z_std == pytest.approx(1.0, abs=0.02)


def test_evaluate_calibration_reports_inflated_z_std_for_overconfident_sigma() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    result = evaluate_calibration(real, mu, sigma / 4.0)

    assert result.z_std == pytest.approx(4.0, abs=0.2)


def test_evaluate_calibration_exposes_the_zscore_it_scored() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    result = evaluate_calibration(real, mu, sigma)

    np.testing.assert_array_equal(result.zscore, learned_zscore(real, mu, sigma))


def test_evaluate_calibration_summary_holds_one_entry_per_coverage_level() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    summary = evaluate_calibration(
        real, mu, sigma, coverage_levels=(1.0, 2.0)
    ).summary()

    assert set(summary) == {
        "nll",
        "ence",
        "z_mean",
        "z_std",
        "error_sigma_spearman",
        "sharpness",
        "dispersion",
        "coverage_1s",
        "coverage_error_1s",
        "coverage_2s",
        "coverage_error_2s",
    }


def test_evaluate_calibration_summary_values_are_plain_floats() -> None:
    real, mu, sigma = _heteroscedastic_pair()

    summary = evaluate_calibration(real, mu, sigma).summary()

    assert all(isinstance(value, float) for value in summary.values())


def test_structural_zscore_matches_direct_division_with_no_smoothing() -> None:
    rng = np.random.default_rng(0)
    structural_delta = rng.uniform(0.0, 1.0, size=_SIZE).astype(np.float32)
    sigma = rng.uniform(0.05, 0.5, size=_SIZE).astype(np.float32)

    result = structural_zscore(structural_delta, sigma, window_size=1)

    np.testing.assert_allclose(
        result, structural_delta / (sigma + 1e-8), rtol=1e-5, atol=1e-6
    )


def test_structural_zscore_is_nonnegative() -> None:
    rng = np.random.default_rng(1)
    structural_delta = rng.uniform(0.0, 1.0, size=_SIZE).astype(np.float32)
    sigma = rng.uniform(0.01, 0.5, size=_SIZE).astype(np.float32)

    result = structural_zscore(structural_delta, sigma)

    assert np.all(result >= 0.0)


def test_structural_zscore_dampens_high_sigma_regions() -> None:
    # Same structural discordance everywhere, but the model is confident on
    # the left half and uncertain (plausible RGB->IR ambiguity) on the right.
    structural_delta = np.ones(_SIZE, dtype=np.float32)
    sigma = np.full(_SIZE, 0.05, dtype=np.float32)
    sigma[:, _SIZE[1] // 2 :] = 0.5

    result = structural_zscore(structural_delta, sigma)

    confident_side = result[:, : _SIZE[1] // 4].mean()
    uncertain_side = result[:, -_SIZE[1] // 4 :].mean()
    assert confident_side > uncertain_side
