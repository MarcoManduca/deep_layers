"""Training utilities for heteroscedastic (mu, log-variance) NLL models.

Kept separate from ``scripts/trainer.py`` by design: the existing
deterministic architectures (``unet``, ``resunet``, ``attention_unet``,
``efficientnet_unet``) and their checkpoints/behaviour are left untouched.
This module wires up the NLL-variant builders instead, reusing
``scripts.trainer.get_callbacks`` (generic checkpoint/early-stopping/
TensorBoard setup, not specific to any architecture or loss). All four NLL
architectures share the same Gaussian NLL loss (unlike
``scripts.trainer.uses_advanced_loss``, which special-cases
``efficientnet_unet``'s deterministic loss) — there is no NLL counterpart
of the advanced loss.
"""

from pathlib import Path

import tensorflow as tf

from scripts.attention_unet_nll import build_attention_unet_nll
from scripts.config import settings
from scripts.efficientnet_unet_nll import build_efficientnet_unet_nll
from scripts.losses import beta_gaussian_nll_loss, gaussian_nll_loss
from scripts.metrics import MuMAEMetric, MuPSNRMetric, MuSSIMMetric
from scripts.resunet_nll import build_resunet_nll
from scripts.trainer import get_callbacks
from scripts.unet_nll import build_unet_nll

_BUILDERS_NLL = {
    "unet_nll": build_unet_nll,
    "resunet_nll": build_resunet_nll,
    "attention_unet_nll": build_attention_unet_nll,
    "efficientnet_unet_nll": build_efficientnet_unet_nll,
}

# Registered NLL loss constructors, keyed by name. All take the same
# (min_log_var, max_log_var, beta) signature so callers can select one by
# name without caring which extra arguments it actually uses — `beta` is
# ignored by "gaussian_nll" and only meaningful for "beta_nll".
_LOSSES_NLL = {
    "gaussian_nll": lambda min_log_var, max_log_var, beta: gaussian_nll_loss(
        min_log_var=min_log_var, max_log_var=max_log_var
    ),
    "beta_nll": lambda min_log_var, max_log_var, beta: beta_gaussian_nll_loss(
        beta=beta, min_log_var=min_log_var, max_log_var=max_log_var
    ),
}

NLL_LOSSES = tuple(_LOSSES_NLL)

__all__ = [
    "get_callbacks",
    "get_model_nll",
    "compile_model_nll",
    "load_model_nll",
    "NLL_LOSSES",
]


def _get_loss_nll(loss_name: str, min_log_var: float, max_log_var: float, beta: float):
    if loss_name not in _LOSSES_NLL:
        raise ValueError(
            f"Unknown NLL loss '{loss_name}'. Available: {list(_LOSSES_NLL)}"
        )
    return _LOSSES_NLL[loss_name](min_log_var, max_log_var, beta)


def get_model_nll(arch_name: str, **kwargs: object) -> tf.keras.Model:
    """Instantiate a heteroscedastic (mu, log-variance) model by name.

    Parameters
    ----------
    arch_name : str
        One of the registered NLL architectures: ``"unet_nll"``,
        ``"resunet_nll"``, ``"attention_unet_nll"``,
        ``"efficientnet_unet_nll"``.
    **kwargs
        Forwarded to the underlying builder function
        (e.g. ``filters``, ``bottleneck``, ``log_var_min``, ``log_var_max``).

    Returns
    -------
    tf.keras.Model
        Uncompiled model with a 2-channel ``(mu, log_var)`` output.

    Raises
    ------
    ValueError
        If ``arch_name`` is not recognised.
    """
    if arch_name not in _BUILDERS_NLL:
        raise ValueError(
            f"Unknown NLL architecture '{arch_name}'. Available: {list(_BUILDERS_NLL)}"
        )
    return _BUILDERS_NLL[arch_name](**kwargs)


def compile_model_nll(
    model: tf.keras.Model,
    lr: float = settings.LEARNING_RATE,
    min_log_var: float = settings.NLL_LOG_VAR_MIN,
    max_log_var: float = settings.NLL_LOG_VAR_MAX,
    loss_name: str = "gaussian_nll",
    beta: float = 0.5,
) -> tf.keras.Model:
    """Compile a heteroscedastic model with Adam and a heteroscedastic NLL loss.

    Metrics (:class:`~scripts.metrics.MuMAEMetric`,
    :class:`~scripts.metrics.MuSSIMMetric`,
    :class:`~scripts.metrics.MuPSNRMetric`) read only the ``mu`` channel,
    so they stay directly comparable to the deterministic architectures'
    ``mae``/``ssim``/``psnr`` in ``030_evaluation.ipynb``.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled (or previously compiled) model with a 2-channel
        ``(mu, log_var)`` output.
    lr : float
        Initial Adam learning rate.
    min_log_var : float
        Lower clip bound for ``log_var`` inside the loss.
    max_log_var : float
        Upper clip bound for ``log_var`` inside the loss.
    loss_name : str
        Which NLL loss to compile with — one of :data:`NLL_LOSSES`
        (``"gaussian_nll"``, the plain Gaussian NLL; ``"beta_nll"``, the
        beta-weighted variant from Seitzer et al. 2022,
        :func:`scripts.losses.beta_gaussian_nll_loss`).
    beta : float
        Weighting exponent, used only when ``loss_name == "beta_nll"``.

    Returns
    -------
    tf.keras.Model
        Compiled model (modified in-place and returned).
    """
    loss_fn = _get_loss_nll(loss_name, min_log_var, max_log_var, beta)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[MuMAEMetric(), MuSSIMMetric(), MuPSNRMetric()],
    )
    return model


def load_model_nll(
    arch_name: str,
    model_dir: Path = settings.MODELS_DIR,
    lr: float = settings.LEARNING_RATE,
    min_log_var: float = settings.NLL_LOG_VAR_MIN,
    max_log_var: float = settings.NLL_LOG_VAR_MAX,
    loss_name: str = "gaussian_nll",
    beta: float = 0.5,
) -> tf.keras.Model:
    """Load the best checkpoint for an NLL architecture and recompile it.

    Loading with ``compile=False`` avoids custom-object registration
    issues; the model is recompiled with :func:`compile_model_nll`. Pass
    the same ``loss_name``/``beta`` the checkpoint was trained with if the
    reported ``loss`` value (not the ``mae``/``ssim``/``psnr`` metrics,
    which don't depend on the loss) needs to be meaningful — the weights
    themselves load and predict identically regardless.

    Parameters
    ----------
    arch_name : str
        NLL architecture identifier.
    model_dir : Path
        Root directory where checkpoints are stored (same tree as
        ``scripts.trainer.load_model`` — the arch name keeps NLL
        checkpoints in their own subdirectory, e.g.
        ``models/attention_unet_nll/``, so they never collide with the
        deterministic checkpoints).
    lr : float
        Learning rate for recompilation.
    min_log_var : float
        Lower clip bound for ``log_var`` inside the loss.
    max_log_var : float
        Upper clip bound for ``log_var`` inside the loss.
    loss_name : str
        Which NLL loss to recompile with — one of :data:`NLL_LOSSES`.
    beta : float
        Weighting exponent, used only when ``loss_name == "beta_nll"``.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for ``evaluate()`` or ``predict()``.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for ``arch_name``.
    """
    model_path = model_dir / arch_name / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {model_path}. Run training first."
        )
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return compile_model_nll(
        model,
        lr=lr,
        min_log_var=min_log_var,
        max_log_var=max_log_var,
        loss_name=loss_name,
        beta=beta,
    )
