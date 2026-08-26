"""Training utilities: model factory, compilation, callbacks, and checkpoint loading."""

from pathlib import Path

import tensorflow as tf

from scripts.attention_unet import build_attention_unet
from scripts.config import settings
from scripts.efficientnet_unet import build_efficientnet_unet
from scripts.losses import combined_loss
from scripts.metrics import PSNRMetric, SSIMMetric
from scripts.resunet import build_resunet
from scripts.unet import build_unet
from scripts.unet_restormer import build_unet_restormer
from scripts.unet_v2 import build_unet_v2

_BUILDERS = {
    "unet": build_unet,
    "unet_v2": build_unet_v2,
    "unet_restormer": build_unet_restormer,
    "resunet": build_resunet,
    "attention_unet": build_attention_unet,
    "efficientnet_unet": build_efficientnet_unet,
    # Phase-2 (unfrozen-encoder) fine-tuning checkpoint (fixing.md #6,
    # Round 2): same builder as "efficientnet_unet", saved under its own
    # `arch_name` (`models/deterministic/efficientnet_unet_ft/`) so the
    # frozen-encoder baseline is never overwritten. `train_single.py`
    # passes `--kwargs '{"freeze_encoder": false}' --init-from
    # <efficientnet_unet checkpoint>` to build and warm-start this variant.
    "efficientnet_unet_ft": build_efficientnet_unet,
}


def get_model(arch_name: str, **kwargs: object) -> tf.keras.Model:
    """Instantiate a model by architecture name.

    Parameters
    ----------
    arch_name : str
        One of ``"unet"``, ``"unet_v2"``, ``"unet_restormer"``,
        ``"resunet"``, ``"attention_unet"``, ``"efficientnet_unet"``,
        ``"efficientnet_unet_ft"`` (the Round 2 two-phase fine-tuning
        checkpoint — same builder as ``"efficientnet_unet"``, see
        ``_BUILDERS``).
    **kwargs
        Forwarded to the underlying builder function
        (e.g. ``filters``, ``bottleneck``).

    Returns
    -------
    tf.keras.Model
        Uncompiled model.

    Raises
    ------
    ValueError
        If ``arch_name`` is not recognised.
    """
    if arch_name not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. Available: {list(_BUILDERS)}"
        )
    return _BUILDERS[arch_name](**kwargs)


def compile_model(
    model: tf.keras.Model,
    arch_name: str,
    lr: float = settings.LEARNING_RATE,
    loss_alpha: float = settings.LOSS_ALPHA,
    weight_decay: float = settings.WEIGHT_DECAY,
    clipvalue: float = settings.GRADIENT_CLIP_VALUE,
) -> tf.keras.Model:
    """Compile a model with Adam and the unified ``combined_loss``.

    Every deterministic architecture — including ``efficientnet_unet``/
    ``efficientnet_unet_ft`` since Round 2 (``fixing.md`` #9) — now trains
    with the same :func:`scripts.losses.combined_loss`
    (``0.16 * Charbonnier + 0.84 * (1 - MS-SSIM)``). Before Round 2,
    ``efficientnet_unet`` used a separate ``combined_loss_advanced`` (MAE +
    Laplacian pyramid + FFT, no perceptual term); that function is still
    defined and unit-tested in ``scripts/losses.py`` but is no longer
    selected by any architecture here — dropping its Laplacian-pyramid/FFT
    terms in favor of a shared perceptual (MS-SSIM) loss is a deliberate
    trade-off documented in ``fixing.md`` §2/§4, not an oversight.
    ``arch_name`` is kept as a parameter (rather than dropped) so callers
    don't need to change, and so a future architecture-specific loss could
    still be reintroduced here without touching every call site.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled (or previously compiled) model.
    arch_name : str
        Architecture identifier (unused for loss selection now that every
        architecture shares ``combined_loss``; kept for API stability).
    lr : float
        Initial Adam learning rate.
    loss_alpha : float
        Charbonnier weight in :func:`scripts.losses.combined_loss`.
    weight_decay : float
        L2 weight decay passed to ``Adam`` (``fixing.md`` #2).
    clipvalue : float
        Per-element gradient clip passed to ``Adam(clipvalue=...)`` —
        guards against ``ms_ssim_loss``'s gradient singularity at a
        collapsed-to-zero scale term (see ``settings.GRADIENT_CLIP_VALUE``,
        ``fixing.md`` §7).

    Returns
    -------
    tf.keras.Model
        Compiled model (modified in-place and returned).
    """
    loss_fn = combined_loss(alpha=loss_alpha)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr, weight_decay=weight_decay, clipvalue=clipvalue
        ),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            SSIMMetric(),
            PSNRMetric(),
        ],
    )
    return model


def get_callbacks(
    arch_name: str,
    log_dir: Path,
    model_dir: Path,
    early_stopping_patience: int = settings.EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = settings.EARLY_STOPPING_MIN_DELTA,
    early_stopping_restore_best_weights: bool = (
        settings.EARLY_STOPPING_RESTORE_BEST_WEIGHTS
    ),
    reduce_lr_factor: float = settings.REDUCE_LR_FACTOR,
    reduce_lr_patience: int = settings.REDUCE_LR_PATIENCE,
    reduce_lr_cooldown: int = settings.REDUCE_LR_COOLDOWN,
    reduce_lr_min_delta: float = settings.REDUCE_LR_MIN_DELTA,
    reduce_lr_min_lr: float = settings.REDUCE_LR_MIN_LR,
) -> list[tf.keras.callbacks.Callback]:
    """Return standard training callbacks for a given architecture.

    Creates ``model_dir / arch_name /`` if it does not exist. All three of
    ``ModelCheckpoint``/``EarlyStopping``/``ReduceLROnPlateau`` monitor
    ``val_loss`` uniformly (``fixing.md``'s callback-tuning note): the
    checkpoint saved is always the one the stop/LR decisions were actually
    based on, rather than splitting onto different metrics that can
    diverge epoch to epoch.

    Parameters
    ----------
    arch_name : str
        Architecture identifier used for naming paths.
    log_dir : Path
        Root directory for TensorBoard event files.
    model_dir : Path
        Root directory for ``.keras`` checkpoints.
    early_stopping_patience : int
        Epochs with no ``val_loss`` improvement before stopping.
    early_stopping_min_delta : float
        Minimum ``val_loss`` change to count as an improvement.
    early_stopping_restore_best_weights : bool
        Whether to restore the best-epoch weights in memory when training
        stops. Safe to leave ``False`` as long as evaluation reloads the
        checkpoint from disk instead of reusing the in-memory model.
    reduce_lr_factor : float
        Multiplicative factor applied to the learning rate on plateau.
    reduce_lr_patience : int
        Epochs with no ``val_loss`` improvement before reducing the
        learning rate.
    reduce_lr_cooldown : int
        Epochs to wait after a reduction before resuming plateau
        monitoring, avoiding rapid repeated reductions.
    reduce_lr_min_delta : float
        Minimum ``val_loss`` change to count as an improvement.
    reduce_lr_min_lr : float
        Lower bound the learning rate is never reduced below.

    Returns
    -------
    list[tf.keras.callbacks.Callback]
        [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard]
    """
    arch_model_dir = model_dir / arch_name
    arch_model_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(arch_model_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=early_stopping_restore_best_weights,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            cooldown=reduce_lr_cooldown,
            min_delta=reduce_lr_min_delta,
            min_lr=reduce_lr_min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir / arch_name),
            histogram_freq=0,
        ),
    ]


def load_model(
    arch_name: str,
    model_dir: Path = settings.MODELS_DIR,
    lr: float = settings.LEARNING_RATE,
    loss_alpha: float = settings.LOSS_ALPHA,
    weight_decay: float = settings.WEIGHT_DECAY,
) -> tf.keras.Model:
    """Load the best checkpoint for an architecture and recompile it.

    Loading with ``compile=False`` avoids custom-object registration
    issues; the model is recompiled with :func:`compile_model`, i.e. the
    same unified ``combined_loss`` every architecture now trains with.

    Parameters
    ----------
    arch_name : str
        Architecture identifier — any ``_BUILDERS`` key, including
        ``"efficientnet_unet_ft"`` (Round 2 two-phase fine-tuning
        checkpoint, ``fixing.md`` #6).
    model_dir : Path
        Root directory where checkpoints are stored.
    lr : float
        Learning rate for recompilation.
    loss_alpha : float
        Charbonnier weight for recompilation.
    weight_decay : float
        L2 weight decay for recompilation. Irrelevant for
        ``evaluate()``/``predict()``, only affects the optimizer state.

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
    return compile_model(
        model, arch_name, lr=lr, loss_alpha=loss_alpha, weight_decay=weight_decay
    )
