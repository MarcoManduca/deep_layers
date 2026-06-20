"""Unit tests for scripts.config."""

from scripts.config import settings


def test_split_ratios_leave_room_for_test_fold() -> None:
    assert 0.0 < settings.TRAIN_RATIO + settings.VAL_RATIO < 1.0


def test_advanced_loss_weights_sum_to_one() -> None:
    total = settings.ADV_LOSS_ALPHA + settings.ADV_LOSS_BETA + settings.ADV_LOSS_GAMMA
    assert total == 1.0


def test_patch_multiple_is_power_of_two_divisible_by_pooling() -> None:
    assert settings.PATCH_MULTIPLE % 16 == 0


def test_crop_size_defaults_to_disabled() -> None:
    assert settings.CROP_SIZE is None
