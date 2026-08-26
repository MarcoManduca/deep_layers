"""Unit tests for scripts.trainer."""

from pathlib import Path

import pytest
import tensorflow as tf

from scripts.trainer import _BUILDERS, compile_model, get_model, load_model


def test_get_model_raises_on_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_model("does_not_exist")


def test_load_model_raises_when_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        load_model("unet", model_dir=tmp_path)


def test_efficientnet_unet_ft_is_registered_without_building_it() -> None:
    # build_efficientnet_unet(_ft) downloads ImageNet weights; only check
    # registration here, same reasoning as efficientnet_unet's lack of a
    # dedicated builder test (see tests/unit/test_nll.py's _OTHER_BUILDERS
    # comment for the NLL counterpart). fixing.md #6 (Round 2): the phase-2
    # fine-tuning checkpoint shares efficientnet_unet's builder, keyed under
    # its own arch_name so get_callbacks/load_model give it its own
    # checkpoint directory (models/deterministic/efficientnet_unet_ft/).
    assert "efficientnet_unet_ft" in _BUILDERS
    assert _BUILDERS["efficientnet_unet_ft"] is _BUILDERS["efficientnet_unet"]


def test_dilated_variants_are_registered() -> None:
    # fixing.md #7 (Round 3): unet_dilated/unet_v2_dilated build a full
    # tf.keras.Model (downloads nothing, unlike efficientnet_unet) —
    # covered directly by tests/unit/test_unet_dilated.py; only check
    # registration and arch_name plumbing here.
    assert "unet_dilated" in _BUILDERS
    assert "unet_v2_dilated" in _BUILDERS


@pytest.mark.parametrize(
    "arch_name",
    [
        "unet",
        "resunet",
        "attention_unet",
        "efficientnet_unet",
        "efficientnet_unet_ft",
        "unet_dilated",
        "unet_v2_dilated",
    ],
)
def test_compile_model_uses_combined_loss_for_every_architecture(
    arch_name: str, dummy_model: tf.keras.Model
) -> None:
    # fixing.md #9 (Round 2): efficientnet_unet no longer gets the separate
    # combined_loss_advanced (MAE + Laplacian + FFT) — every deterministic
    # architecture, including the EfficientNet ones, now shares the same
    # unified combined_loss.
    compile_model(dummy_model, arch_name)
    assert dummy_model.loss.__name__ == "combined_loss"
