"""Unit tests for scripts.trainer."""

from pathlib import Path

import pytest
import tensorflow as tf

from scripts.trainer import (
    compile_model,
    get_model,
    load_model,
    uses_advanced_loss,
)


@pytest.mark.parametrize(
    "arch_name, expected",
    [
        ("efficientnet_unet", True),
        ("unet", False),
        ("resunet", False),
        ("attention_unet", False),
    ],
)
def test_uses_advanced_loss_only_for_efficientnet(
    arch_name: str, expected: bool
) -> None:
    assert uses_advanced_loss(arch_name) is expected


def test_get_model_raises_on_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_model("does_not_exist")


def test_load_model_raises_when_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        load_model("unet", model_dir=tmp_path)


def test_compile_model_uses_advanced_loss_for_efficientnet(
    dummy_model: tf.keras.Model,
) -> None:
    compile_model(dummy_model, "efficientnet_unet")
    assert dummy_model.loss.__name__ == "combined_loss_advanced"


def test_compile_model_uses_combined_loss_for_plain_unet(
    dummy_model: tf.keras.Model,
) -> None:
    compile_model(dummy_model, "unet")
    assert dummy_model.loss.__name__ == "combined_loss"
