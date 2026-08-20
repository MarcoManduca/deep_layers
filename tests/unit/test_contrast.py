"""Unit tests for scripts.contrast."""

import numpy as np
import pytest

from scripts.contrast import ZScale, ZScaleMode


def _ramp() -> np.ndarray:
    """Signed z-score map spanning ``[-10, 10]``."""
    return np.linspace(-10.0, 10.0, 401, dtype=np.float32)


# --- limit selection ---------------------------------------------------------


def test_default_scale_keeps_the_raw_zscore_in_z_units() -> None:
    z = _ramp()

    scaled = ZScale().apply(z)

    np.testing.assert_array_equal(scaled.values, z)
    assert (scaled.vmin, scaled.vmax) == (-4.0, 4.0)


def test_fixed_mode_ignores_the_data_distribution() -> None:
    quiet = ZScale().limit(np.zeros(10, dtype=np.float32))
    loud = ZScale().limit(_ramp() * 100.0)

    assert quiet == loud == 4.0


def test_percentile_mode_derives_the_limit_from_the_map() -> None:
    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=100.0)

    assert scale.limit(_ramp()) == pytest.approx(10.0)


def test_percentile_mode_falls_back_to_vmax_on_a_degenerate_map() -> None:
    scale = ZScale(mode=ZScaleMode.PERCENTILE, vmax=4.0)

    assert scale.limit(np.zeros((8, 8), dtype=np.float32)) == 4.0


def test_percentile_mode_raises_without_a_map_to_measure() -> None:
    with pytest.raises(ValueError, match="at least one z-score map"):
        ZScale(mode=ZScaleMode.PERCENTILE).limit()


def test_apply_honours_an_explicit_shared_limit() -> None:
    scaled = ZScale(mode=ZScaleMode.PERCENTILE).apply(_ramp(), limit=2.5)

    assert (scaled.vmin, scaled.vmax) == (-2.5, 2.5)
    assert scaled.limit == 2.5


# --- gamma compression -------------------------------------------------------


def test_gamma_below_one_lifts_faint_values() -> None:
    z = np.array([2.0], dtype=np.float32)

    linear = ZScale(vmax=4.0).apply(z)
    lifted = ZScale(vmax=4.0, gamma=0.5).apply(z)

    assert linear.values[0] / linear.vmax == pytest.approx(0.5)
    assert lifted.values[0] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_gamma_above_one_suppresses_faint_values() -> None:
    z = np.array([2.0], dtype=np.float32)

    scaled = ZScale(vmax=4.0, gamma=2.0).apply(z)

    assert scaled.values[0] == pytest.approx(0.25, abs=1e-6)


def test_gamma_output_is_bounded_and_sign_preserving() -> None:
    z = _ramp()

    scaled = ZScale(vmax=4.0, gamma=0.5).apply(z)

    assert scaled.values.min() >= -1.0
    assert scaled.values.max() <= 1.0
    np.testing.assert_array_equal(np.sign(scaled.values), np.sign(z))


def test_gamma_scaled_values_use_a_unit_display_range() -> None:
    scaled = ZScale(vmax=4.0, gamma=0.5).apply(_ramp())

    assert (scaled.vmin, scaled.vmax) == (-1.0, 1.0)
    assert scaled.limit == 4.0


# --- cross-model scaling -----------------------------------------------------


def test_apply_many_scales_every_map_against_one_shared_limit() -> None:
    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=100.0)
    maps = {"unet_nll": _ramp(), "resunet_nll": _ramp() / 2.0}

    scaled, _ = scale.apply_many(maps)

    shared = scale.limit(*maps.values())
    np.testing.assert_allclose(
        scaled["resunet_nll"], scale.apply(maps["resunet_nll"], limit=shared).values
    )


def test_apply_many_returns_the_vrange_for_plot_signal_comparison() -> None:
    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=100.0)

    _, vrange = scale.apply_many({"unet_nll": _ramp()})

    assert vrange == (-10.0, 10.0)


def test_apply_many_returns_a_unit_vrange_under_gamma() -> None:
    _, vrange = ZScale(gamma=0.5).apply_many({"unet_nll": _ramp()})

    assert vrange == (-1.0, 1.0)


def test_apply_many_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one map"):
        ZScale().apply_many({})


# --- labelling and validation ------------------------------------------------


def test_label_reports_a_fixed_limit_without_extra_annotation() -> None:
    assert ZScale().apply(_ramp()).label == "|z| <= 4.00"


def test_label_reports_the_percentile_and_gamma_used() -> None:
    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=99.0, gamma=0.5)

    label = scale.apply(_ramp()).label

    assert "(p99)" in label
    assert "gamma=0.5" in label


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vmax": 0.0},
        {"vmax": -1.0},
        {"gamma": 0.0},
        {"gamma": -0.5},
        {"percentile": 0.0},
        {"percentile": 101.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ZScale(**kwargs)
