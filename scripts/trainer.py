"""Training utilities: model factory, compilation, callbacks, and checkpoint loading."""

from pathlib import Path

import tensorflow as tf

from scripts.attention_unet import build_attention_unet
from scripts.efficientnet_unet import build_efficientnet_unet
from scripts.losses import combined_loss, combined_loss_advanced
from scripts.metrics import PSNRMetric, SSIMMetric
from scripts.resunet import build_resunet
from scripts.unet import build_unet

_BUILDERS = {
    "unet": build_unet,
    "resunet": build_resunet,
    "attention_unet": build_attention_unet,
    "efficientnet_unet": build_efficientnet_unet,
}


def get_model(arch_name: str, **kwargs: object) -> tf.keras.Model:
    """Instantiate a model by architecture name.

    Parameters
    ----------
    arch_name : str
        One of ``"unet"``, ``"resunet"``, ``"attention_unet"``,
        ``"efficientnet_unet"``.
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
            f"Unknown architecture '{arch_name}'. "
            f"Available: {list(_BUILDERS)}"
        )
    return _BUILDERS[arch_name](**kwargs)


def compile_model(
    model: tf.keras.Model,
    lr: float = 1e-4,
    loss_alpha: float = 0.7,
    advanced_loss: bool = False,
) -> tf.keras.Model:
    """Compile a model with Adam and the selected loss function.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled (or previously compiled) model.
    lr : float
        Initial Adam learning rate.
    loss_alpha : float
        MAE weight in the combined loss.  Ignored when
        ``advanced_loss=True``.
    advanced_loss : bool
        If ``True``, use :func:`scripts.losses.combined_loss_advanced`
        (MAE + Laplacian + FFT).  Recommended for
        ``"efficientnet_unet"``.  Defaults to ``False``.

    Returns
    -------
    tf.keras.Model
        Compiled model (modified in-place and returned).
    """
    loss_fn = (
        combined_loss_advanced() if advanced_loss else combined_loss(alpha=loss_alpha)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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
) -> list[tf.keras.callbacks.Callback]:
    """Return standard training callbacks for a given architecture.

    Creates ``model_dir / arch_name /`` if it does not exist.

    Parameters
    ----------
    arch_name : str
        Architecture identifier used for naming paths.
    log_dir : Path
        Root directory for TensorBoard event files.
    model_dir : Path
        Root directory for ``.keras`` checkpoints.

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
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir / arch_name),
            histogram_freq=0,
        ),
    ]


def load_model(
    arch_name: str,
    model_dir: Path,
    lr: float = 1e-4,
    loss_alpha: float = 0.7,
    advanced_loss: bool = False,
) -> tf.keras.Model:
    """Load the best checkpoint for an architecture and recompile it.

    Loading with ``compile=False`` avoids custom-object registration
    issues; the model is recompiled with the same loss and metrics used
    during training.

    Parameters
    ----------
    arch_name : str
        Architecture identifier.
    model_dir : Path
        Root directory where checkpoints are stored.
    lr : float
        Learning rate for recompilation.
    loss_alpha : float
        MAE weight for recompilation.  Ignored when
        ``advanced_loss=True``.
    advanced_loss : bool
        Must match the value used at training time.

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
            f"No checkpoint found at {model_path}. "
            "Run training first."
        )
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return compile_model(model, lr=lr, loss_alpha=loss_alpha, advanced_loss=advanced_loss)
