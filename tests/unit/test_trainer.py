"""Unit tests for scripts.trainer."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts.trainer import (
    LossName,
    build_loss,
    compile_model,
    get_loss_name,
    get_model,
    load_model,
)


@pytest.mark.parametrize(
    "arch_name, expected",
    [
        ("unet", LossName.COMBINED),
        ("resunet", LossName.COMBINED),
        ("attention_unet", LossName.COMBINED),
        ("efficientnet_unet", LossName.ADVANCED),
    ],
)
def test_get_loss_name_returns_registered_loss(
    arch_name: str, expected: LossName
) -> None:
    assert get_loss_name(arch_name) is expected


def test_get_loss_name_raises_on_unregistered_architecture() -> None:
    with pytest.raises(ValueError, match="No loss registered"):
        get_loss_name("does_not_exist")


@pytest.mark.parametrize("loss_name", list(LossName))
def test_build_loss_returns_callable_named_after_the_registry_entry(
    loss_name: LossName,
) -> None:
    assert build_loss(loss_name).__name__ == loss_name.value


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


def test_compile_model_raises_on_unregistered_architecture(
    dummy_model: tf.keras.Model,
) -> None:
    with pytest.raises(ValueError, match="No loss registered"):
        compile_model(dummy_model, "does_not_exist")


def test_normalized_loss_trains_a_step_in_graph_mode(
    dummy_model: tf.keras.Model,
) -> None:
    # The normalised loss reads the static channel count off y_pred to size
    # its depthwise blur; this exercises that path inside a compiled
    # train_step, where shapes are only partially known.
    dummy_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_loss(LossName.NORMALIZED),
    )
    rng = np.random.default_rng(0)
    rgb = rng.random((2, 32, 32, 3)).astype(np.float32)
    ir = rng.random((2, 32, 32, 1)).astype(np.float32)

    history = dummy_model.fit(rgb, ir, epochs=1, batch_size=2, verbose=0)

    assert np.isfinite(history.history["loss"][0])
