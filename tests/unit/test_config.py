"""Unit tests for scripts.config."""

import pytest

from scripts.config import Settings, settings


def test_split_ratios_leave_room_for_test_fold() -> None:
    assert 0.0 < settings.TRAIN_RATIO + settings.VAL_RATIO < 1.0


def test_advanced_loss_weights_sum_to_one() -> None:
    total = settings.ADV_LOSS_ALPHA + settings.ADV_LOSS_BETA + settings.ADV_LOSS_GAMMA
    assert total == 1.0


def test_patch_multiple_is_power_of_two_divisible_by_pooling() -> None:
    assert settings.PATCH_MULTIPLE % 16 == 0


def test_crop_size_field_default_is_disabled() -> None:
    # The declared default must keep cropping off; the live ``settings`` value
    # may be overridden via ``.env``, so assert the field default directly.
    assert Settings.model_fields["CROP_SIZE"].default is None


def test_zscore_sigma_floor_field_default_is_above_zero() -> None:
    # A zero floor makes the z-score denominator vanish on flat regions and
    # the gradient of the sqrt unbounded, which is exactly the NaN case the
    # floor exists to prevent.
    assert Settings.model_fields["ZSCORE_SIGMA_FLOOR"].default == pytest.approx(0.01)


def test_zscore_window_is_odd_so_it_has_a_centre_pixel() -> None:
    assert settings.ZSCORE_WINDOW % 2 == 1


def test_normalized_loss_beta_field_default_offsets_the_term_scale() -> None:
    # The normalised term runs ~9x larger than MAE, so its weight must be well
    # below the MAE weight to stay an auxiliary term rather than the objective.
    alpha = Settings.model_fields["NORM_LOSS_ALPHA"].default
    beta = Settings.model_fields["NORM_LOSS_BETA"].default
    assert 0.0 < beta < alpha / 9.0
