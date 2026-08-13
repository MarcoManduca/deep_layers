"""Unit tests for the heteroscedastic (mu, log-variance) NLL pipeline:
scripts.attention_unet_nll, scripts.losses.gaussian_nll_loss,
scripts.metrics.Mu*Metric, scripts.trainer_nll, and
scripts.inference_utils_nll.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts.attention_unet_nll import build_attention_unet_nll
from scripts.inference_utils_nll import predict_with_overlap_nll
from scripts.losses import beta_gaussian_nll_loss, gaussian_nll_loss
from scripts.metrics import MuMAEMetric, MuPSNRMetric, MuSSIMMetric
from scripts.resunet_nll import build_resunet_nll
from scripts.trainer_nll import (
    compile_model_nll,
    get_model_nll,
    load_model_nll,
)
from scripts.unet_nll import build_unet_nll

_TINY = {"filters": [8, 16], "bottleneck": 32}

# (builder, kwargs, expected model name) for the plain-conv-block NLL
# architectures. efficientnet_unet_nll is excluded: it downloads
# ImageNet-pretrained weights, which build_efficientnet_unet (its
# deterministic counterpart) is likewise never unit-tested against, for
# the same reason. attention_unet_nll keeps its own dedicated tests below
# (attention-gate-specific coverage predates this list).
_OTHER_BUILDERS = [
    (build_unet_nll, _TINY, "unet_nll"),
    (build_resunet_nll, _TINY, "resunet_nll"),
]


def _batch(seed: int, channels: int = 1) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    return tf.constant(rng.random((2, 16, 16, channels)), dtype=tf.float32)


# ---------------------------------------------------------------------------
# scripts.attention_unet_nll
# ---------------------------------------------------------------------------


def test_builder_outputs_two_channels() -> None:
    # Arrange
    model = build_attention_unet_nll(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()

    # Assert
    assert y.shape == (1, 16, 16, 2)


def test_builder_mu_channel_is_in_unit_range() -> None:
    # Arrange
    model = build_attention_unet_nll(**_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()
    mu = y[..., 0]

    # Assert
    assert mu.min() >= 0.0
    assert mu.max() <= 1.0


def test_builder_log_var_channel_is_clipped() -> None:
    # Arrange
    model = build_attention_unet_nll(log_var_min=-2.0, log_var_max=2.0, **_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()
    log_var = y[..., 1]

    # Assert
    assert log_var.min() >= -2.0
    assert log_var.max() <= 2.0


def test_builder_preserves_non_square_spatial_dimensions() -> None:
    # Arrange
    model = build_attention_unet_nll(**_TINY)
    x = np.random.rand(1, 32, 48, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()

    # Assert
    assert y.shape == (1, 32, 48, 2)


def test_builder_survives_a_save_load_round_trip(tmp_path: Path) -> None:
    # Regression test: the log_var clip used to be a Lambda layer, which
    # Keras refuses to deserialize by default (arbitrary-code-execution
    # guard against Python-lambda Lambda layers), breaking
    # tf.keras.models.load_model on any saved checkpoint. ClipLogVar (a
    # named, serializable layer) fixes this — this test would fail on the
    # old Lambda-based implementation.
    model = build_attention_unet_nll(**_TINY)
    ckpt = tmp_path / "model.keras"
    model.save(str(ckpt))

    reloaded = tf.keras.models.load_model(str(ckpt), compile=False)

    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y = reloaded(x, training=False).numpy()
    assert y.shape == (1, 16, 16, 2)


# ---------------------------------------------------------------------------
# scripts.unet_nll / scripts.resunet_nll (shared coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder,kwargs,expected_name", _OTHER_BUILDERS)
def test_other_builder_outputs_two_channels(builder, kwargs, expected_name) -> None:
    model = builder(**kwargs)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    y = model(x, training=False).numpy()

    assert y.shape == (1, 16, 16, 2)
    assert model.name == expected_name


@pytest.mark.parametrize("builder,kwargs,expected_name", _OTHER_BUILDERS)
def test_other_builder_mu_channel_is_in_unit_range(
    builder, kwargs, expected_name
) -> None:
    model = builder(**kwargs)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    mu = model(x, training=False).numpy()[..., 0]

    assert mu.min() >= 0.0
    assert mu.max() <= 1.0


@pytest.mark.parametrize("builder,kwargs,expected_name", _OTHER_BUILDERS)
def test_other_builder_log_var_channel_is_clipped(
    builder, kwargs, expected_name
) -> None:
    model = builder(log_var_min=-2.0, log_var_max=2.0, **kwargs)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    log_var = model(x, training=False).numpy()[..., 1]

    assert log_var.min() >= -2.0
    assert log_var.max() <= 2.0


@pytest.mark.parametrize("builder,kwargs,expected_name", _OTHER_BUILDERS)
def test_other_builder_survives_a_save_load_round_trip(
    builder, kwargs, expected_name, tmp_path: Path
) -> None:
    model = builder(**kwargs)
    ckpt = tmp_path / "model.keras"
    model.save(str(ckpt))

    reloaded = tf.keras.models.load_model(str(ckpt), compile=False)

    x = np.random.rand(1, 16, 16, 3).astype("float32")
    y = reloaded(x, training=False).numpy()
    assert y.shape == (1, 16, 16, 2)


# ---------------------------------------------------------------------------
# scripts.losses.gaussian_nll_loss
# ---------------------------------------------------------------------------


def test_gaussian_nll_loss_is_low_for_accurate_confident_prediction() -> None:
    # mu == y_true and low log_var (confident, correct) -> loss near its
    # minimum (0.5 * log_var, since the squared-error term vanishes).
    yt = _batch(0, channels=1)
    mu = yt
    log_var = tf.fill(yt.shape, -6.0)
    y_pred = tf.concat([mu, log_var], axis=-1)

    loss_val = float(gaussian_nll_loss()(yt, y_pred))
    assert loss_val == pytest.approx(-3.0, abs=1e-4)


def test_gaussian_nll_loss_penalises_overconfident_wrong_prediction() -> None:
    # mu far from y_true but log_var very negative (falsely confident)
    # should cost much more than the same mu error with high log_var
    # (honestly uncertain).
    yt = tf.ones((2, 8, 8, 1), dtype=tf.float32) * 0.9
    mu = tf.zeros((2, 8, 8, 1), dtype=tf.float32)

    overconfident = tf.concat([mu, tf.fill(yt.shape, -6.0)], axis=-1)
    uncertain = tf.concat([mu, tf.fill(yt.shape, 6.0)], axis=-1)

    loss_fn = gaussian_nll_loss()
    assert float(loss_fn(yt, overconfident)) > float(loss_fn(yt, uncertain))


def test_gaussian_nll_loss_clips_log_var_before_use() -> None:
    yt = _batch(0, channels=1)
    mu = yt
    unclipped_log_var = tf.fill(yt.shape, 1000.0)
    y_pred = tf.concat([mu, unclipped_log_var], axis=-1)

    loss_val = float(gaussian_nll_loss(min_log_var=-6.0, max_log_var=6.0)(yt, y_pred))
    assert np.isfinite(loss_val)
    assert loss_val == pytest.approx(3.0, abs=1e-4)


def test_gaussian_nll_loss_sets_keras_friendly_name() -> None:
    assert gaussian_nll_loss().__name__ == "gaussian_nll_loss"


# ---------------------------------------------------------------------------
# scripts.losses.beta_gaussian_nll_loss
# ---------------------------------------------------------------------------


def test_beta_nll_loss_matches_plain_nll_when_beta_is_zero() -> None:
    # weight = sigma^(2*0) = 1 everywhere -> identical to gaussian_nll_loss.
    yt = tf.ones((2, 8, 8, 1), dtype=tf.float32) * 0.9
    mu = tf.zeros((2, 8, 8, 1), dtype=tf.float32)
    log_var = tf.fill(yt.shape, -1.0)
    y_pred = tf.concat([mu, log_var], axis=-1)

    plain = float(gaussian_nll_loss()(yt, y_pred))
    beta0 = float(beta_gaussian_nll_loss(beta=0.0)(yt, y_pred))
    assert beta0 == pytest.approx(plain, abs=1e-5)


def test_beta_nll_loss_gives_high_variance_pixels_more_gradient_than_plain() -> None:
    # Two pixels, same squared error, different sigma (low_var vs. high_var
    # log_var). Plain NLL's d(loss)/d(mu) shrinks as sigma^-2; beta=0.5's
    # shrinks only as sigma^-1 (Seitzer et al. 2022) — so the high-variance
    # pixel's gradient share of the total should be larger under beta-NLL
    # than under plain NLL.
    yt = tf.constant([[[[0.9], [0.9]]]], dtype=tf.float32)  # shape (1, 1, 2, 1)
    log_var = tf.constant([[[[-2.0], [2.0]]]], dtype=tf.float32)

    def grad_ratio(loss_fn) -> float:
        mu = tf.Variable(tf.zeros_like(yt))
        with tf.GradientTape() as tape:
            y_pred = tf.concat([mu, log_var], axis=-1)
            loss = loss_fn(yt, y_pred)
        (grad,) = tape.gradient(loss, [mu])
        low_var_grad = abs(float(grad[0, 0, 0, 0]))
        high_var_grad = abs(float(grad[0, 0, 1, 0]))
        return high_var_grad / low_var_grad

    plain_ratio = grad_ratio(gaussian_nll_loss())
    beta_ratio = grad_ratio(beta_gaussian_nll_loss(beta=0.5))
    assert beta_ratio > plain_ratio


def test_beta_nll_loss_clips_log_var_before_use() -> None:
    yt = _batch(0, channels=1)
    mu = yt
    unclipped_log_var = tf.fill(yt.shape, 1000.0)
    y_pred = tf.concat([mu, unclipped_log_var], axis=-1)

    loss_val = float(
        beta_gaussian_nll_loss(beta=0.5, min_log_var=-6.0, max_log_var=6.0)(yt, y_pred)
    )
    assert np.isfinite(loss_val)


def test_beta_nll_loss_sets_keras_friendly_name() -> None:
    assert beta_gaussian_nll_loss().__name__ == "beta_gaussian_nll_loss"


# ---------------------------------------------------------------------------
# scripts.metrics.Mu*Metric
# ---------------------------------------------------------------------------


def _two_channel_pred(mu_seed: int, log_var_value: float = 0.0) -> tf.Tensor:
    mu = _batch(mu_seed, channels=1)
    log_var = tf.fill(mu.shape, log_var_value)
    return tf.concat([mu, log_var], axis=-1)


def test_mu_ssim_metric_is_one_when_mu_matches_ground_truth() -> None:
    yt = _batch(0, channels=1)
    y_pred = tf.concat([yt, tf.zeros_like(yt)], axis=-1)

    metric = MuSSIMMetric()
    metric.update_state(yt, y_pred)
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)


def test_mu_psnr_metric_ignores_log_var_channel() -> None:
    yt = _batch(0, channels=1)
    pred_a = tf.concat([yt, tf.zeros_like(yt)], axis=-1)
    pred_b = tf.concat([yt, tf.fill(yt.shape, 5.0)], axis=-1)

    metric_a, metric_b = MuPSNRMetric(), MuPSNRMetric()
    metric_a.update_state(yt, pred_a)
    metric_b.update_state(yt, pred_b)
    assert float(metric_a.result()) == pytest.approx(float(metric_b.result()))


def test_mu_mae_metric_matches_manual_computation() -> None:
    yt = _batch(0, channels=1)
    y_pred = _two_channel_pred(mu_seed=1)
    mu = y_pred[..., 0:1]

    metric = MuMAEMetric()
    metric.update_state(yt, y_pred)
    expected = float(tf.reduce_mean(tf.abs(yt - mu)))
    assert float(metric.result()) == pytest.approx(expected, rel=1e-5)


def test_mu_metric_reset_state_clears_accumulators() -> None:
    yt = _batch(0, channels=1)
    metric = MuSSIMMetric()
    metric.update_state(yt, _two_channel_pred(mu_seed=1))

    metric.reset_state()
    y_pred_match = tf.concat([yt, tf.zeros_like(yt)], axis=-1)
    metric.update_state(yt, y_pred_match)
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# scripts.trainer_nll
# ---------------------------------------------------------------------------


def test_get_model_nll_raises_on_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown NLL architecture"):
        get_model_nll("does_not_exist")


def test_get_model_nll_builds_registered_architecture() -> None:
    model = get_model_nll("attention_unet_nll", **_TINY)
    assert model.name == "attention_unet_nll"


@pytest.mark.parametrize("arch_name", ["unet_nll", "resunet_nll", "attention_unet_nll"])
def test_get_model_nll_builds_each_plain_registered_architecture(
    arch_name: str,
) -> None:
    model = get_model_nll(arch_name, **_TINY)
    assert model.name == arch_name


def test_efficientnet_unet_nll_is_registered_without_building_it() -> None:
    # build_efficientnet_unet_nll downloads ImageNet weights; only check
    # registration here, mirroring efficientnet_unet's lack of a
    # dedicated builder test (see _OTHER_BUILDERS above).
    from scripts.trainer_nll import _BUILDERS_NLL

    assert "efficientnet_unet_nll" in _BUILDERS_NLL


def test_load_model_nll_raises_when_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        load_model_nll("attention_unet_nll", model_dir=tmp_path)


def test_compile_model_nll_uses_gaussian_nll_loss(
    dummy_model_nll: tf.keras.Model,
) -> None:
    compile_model_nll(dummy_model_nll)
    assert dummy_model_nll.loss.__name__ == "gaussian_nll_loss"


def test_compile_model_nll_forward_pass_produces_finite_loss(
    dummy_model_nll: tf.keras.Model,
) -> None:
    compile_model_nll(dummy_model_nll)
    x = _batch(0, channels=3)
    yt = _batch(0, channels=1)
    metrics = dummy_model_nll.evaluate(x, yt, verbose=0, return_dict=True)
    assert np.isfinite(metrics["loss"])


def test_compile_model_nll_uses_beta_nll_loss(
    dummy_model_nll: tf.keras.Model,
) -> None:
    compile_model_nll(dummy_model_nll, loss_name="beta_nll", beta=0.5)
    assert dummy_model_nll.loss.__name__ == "beta_gaussian_nll_loss"


def test_compile_model_nll_raises_on_unknown_loss_name(
    dummy_model_nll: tf.keras.Model,
) -> None:
    with pytest.raises(ValueError, match="Unknown NLL loss"):
        compile_model_nll(dummy_model_nll, loss_name="does_not_exist")


# ---------------------------------------------------------------------------
# scripts.inference_utils_nll
# ---------------------------------------------------------------------------


def test_predict_with_overlap_nll_rejects_patch_size_not_multiple_of_16(
    dummy_model_nll: tf.keras.Model,
) -> None:
    image = np.random.rand(48, 48, 3).astype("float32")
    with pytest.raises(ValueError, match="multiple of 16"):
        predict_with_overlap_nll(dummy_model_nll, image, patch_size=30)


def test_predict_with_overlap_nll_rejects_stride_larger_than_patch(
    dummy_model_nll: tf.keras.Model,
) -> None:
    image = np.random.rand(48, 48, 3).astype("float32")
    with pytest.raises(ValueError, match="must be ≤ patch_size"):
        predict_with_overlap_nll(dummy_model_nll, image, patch_size=16, stride=32)


def test_predict_with_overlap_nll_returns_two_channels_at_input_resolution(
    dummy_model_nll: tf.keras.Model,
) -> None:
    # Arrange
    image = np.random.rand(40, 50, 3).astype("float32")

    # Act
    pred = predict_with_overlap_nll(dummy_model_nll, image, patch_size=16, stride=8)

    # Assert
    assert pred.shape == (40, 50, 2)
