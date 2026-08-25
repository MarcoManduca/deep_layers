"""ResUNet with a heteroscedastic (mu, log-scale) output head.

New architecture, kept separate from ``resunet.py``: it reuses the same
residual encoder/decoder structure, but the final ``1x1`` conv is
duplicated into two parallel branches instead of a single deterministic
prediction — a mean ``mu`` (same role as the existing model's output) and
a second channel (named/clipped as ``log_var`` for historical reasons —
see ``scripts.config.Settings.NLL_LOG_VAR_MIN``) that is the model's own
learned estimate of how ambiguous the local RGB context has historically
been. Trained with :func:`scripts.losses.laplace_nll_loss` via
``scripts/trainer_nll.py`` (``fixing.md`` #10 — this channel is a Laplace
log-scale, not a Gaussian log-variance, despite the name).
See ``code-review.md`` §7.6 for the design rationale and references.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.config import settings
from scripts.nll_layers import ClipLogVar
from scripts.norm_utils import num_groups as _num_groups
from scripts.norm_utils import relu as _relu


def _residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Residual block: Conv → GroupNorm → ReLU → Conv → GroupNorm → Add(shortcut) → ReLU.

    A 1×1 projection shortcut is always applied to match channel counts
    (kept unconditional — see ``scripts.resunet._residual_block`` for why
    a conditional identity path would be dead code in this architecture).
    Uses ``GroupNormalization`` instead of ``BatchNormalization``
    (``fixing.md`` #1) and He init instead of Xavier (``fixing.md`` #3).

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of output filters.

    Returns
    -------
    tf.Tensor
        Output feature map with shape ``(..., H, W, filters)``.
    """
    shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    shortcut = layers.GroupNormalization(groups=_num_groups(filters))(shortcut)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
    x = _relu(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)

    x = layers.Add()([x, shortcut])
    x = _relu(x)
    return x


def build_resunet_nll(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    log_var_min: float | None = None,
    log_var_max: float | None = None,
) -> tf.keras.Model:
    """Build a 4-level ResUNet predicting per-pixel ``(mu, log_var)``.

    Identical residual encoder/decoder structure to
    :func:`scripts.resunet.build_resunet`; only the output head differs.
    ``mu`` is the expected IR value (same range/activation as the
    deterministic model); ``log_var`` is the model's own learned estimate
    of aleatoric uncertainty for that prediction, clipped to
    ``[log_var_min, log_var_max]`` for numerical stability before being
    consumed by :func:`scripts.losses.laplace_nll_loss`.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block.
        Defaults to 1024.
    log_var_min : float or None
        Lower clip bound for the ``log_var`` channel. Defaults to
        ``settings.NLL_LOG_VAR_MIN``.
    log_var_max : float or None
        Upper clip bound for the ``log_var`` channel. Defaults to
        ``settings.NLL_LOG_VAR_MAX``.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output shape
        ``(None, None, None, 2)``: channel 0 is ``mu`` (sigmoid, values in
        ``[0, 1]``), channel 1 is ``log_var`` (linear, clipped).
    """
    if filters is None:
        filters = [64, 128, 256, 512]
    if log_var_min is None:
        log_var_min = settings.NLL_LOG_VAR_MIN
    if log_var_max is None:
        log_var_max = settings.NLL_LOG_VAR_MAX

    inputs = layers.Input(shape=(None, None, 3))

    skips: list[tf.Tensor] = []
    x = inputs
    for f in filters:
        x = _residual_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = _residual_block(x, bottleneck)

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _residual_block(x, f)

    mu = layers.Conv2D(1, 1, activation="sigmoid", name="mu")(x)
    log_var = layers.Conv2D(1, 1, activation="linear", name="log_var_raw")(x)
    log_var = ClipLogVar(log_var_min, log_var_max, name="log_var")(log_var)
    outputs = layers.Concatenate(name="mu_log_var")([mu, log_var])

    return tf.keras.Model(inputs, outputs, name="resunet_nll")
