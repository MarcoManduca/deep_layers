"""Central configuration for the deep-layers pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Pipeline hyperparameters and directory paths.

    Parameters
    ----------
    IR_DIR : Path
        Directory containing IR images.
    RGB_DIR : Path
        Directory containing RGB images.
    MODELS_DIR : Path
        Root directory for saved model checkpoints.
    LOGS_DIR : Path
        Root directory for TensorBoard logs.
    PATCH_MULTIPLE : int
        Pad image dimensions to the nearest multiple of this value.
    BATCH_SIZE : int
        Training batch size.
    LEARNING_RATE : float
        Initial Adam learning rate.
    EPOCHS : int
        Maximum training epochs.
    SEED : int
        Global random seed for reproducibility.
    TRAIN_RATIO : float
        Fraction of artworks assigned to the training fold.
    VAL_RATIO : float
        Fraction of artworks assigned to the validation fold.
    MOCKUP_ARTWORK_IDS : list[str]
        Artwork IDs that are synthetic paint-on-support mockups created
        specifically to aid training (not real artworks to generalize to).
        Used by ``dataset.mockup_aware_train_val_test_split`` to keep almost
        all of their sections in train/val instead of holding entire groups
        out for test.
    MOCKUP_TEST_RATIO : float
        Fraction of mockup *pairs* (not groups) sent to test by
        ``dataset.mockup_aware_train_val_test_split`` — the mockup-side
        counterpart of ``TRAIN_RATIO``/``VAL_RATIO``. The remaining pairs
        split into train/val using the same relative proportion as
        ``TRAIN_RATIO``/``VAL_RATIO`` do for real artworks.
    LOSS_ALPHA : float
        Weight of the Charbonnier term in the combined Charbonnier +
        (1 - MS-SSIM) loss (``scripts.losses.combined_loss``).
    CHARBONNIER_EPS : float
        Smoothing constant for the Charbonnier term, see
        ``scripts.losses.charbonnier_loss``.
    WEIGHT_DECAY : float
        L2 weight decay passed to ``Adam(weight_decay=...)`` during
        training (``fixing.md`` #2).
    NLL_BETA : float
        Weighting exponent for the beta-weighted NLL losses
        (``scripts.losses.beta_gaussian_nll_loss``,
        ``scripts.losses.laplace_nll_loss``).
    EARLY_STOPPING_PATIENCE : int
        Epochs with no ``val_loss`` improvement before training stops.
    EARLY_STOPPING_MIN_DELTA : float
        Minimum ``val_loss`` change to count as an improvement for
        early stopping.
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS : bool
        Whether to restore the best-epoch weights in memory when training
        stops. Safe to leave ``False`` as long as evaluation always
        reloads the checkpoint ``ModelCheckpoint`` saved to disk rather
        than reusing the in-memory model right after ``fit()``.
    REDUCE_LR_FACTOR : float
        Multiplicative factor applied to the learning rate on plateau.
    REDUCE_LR_PATIENCE : int
        Epochs with no ``val_loss`` improvement before reducing the
        learning rate.
    REDUCE_LR_COOLDOWN : int
        Epochs to wait after a learning-rate reduction before resuming
        plateau monitoring, avoiding rapid repeated reductions.
    REDUCE_LR_MIN_DELTA : float
        Minimum ``val_loss`` change to count as an improvement for the
        learning-rate scheduler.
    REDUCE_LR_MIN_LR : float
        Lower bound the learning rate is never reduced below.
    ADV_LOSS_ALPHA : float
        Weight of the MAE term in ``combined_loss_advanced``.
    ADV_LOSS_BETA : float
        Weight of the Laplacian pyramid term in ``combined_loss_advanced``.
    ADV_LOSS_GAMMA : float
        Weight of the FFT magnitude term in ``combined_loss_advanced``.
    LAPLACIAN_LEVELS : int
        Number of Laplacian pyramid levels in ``combined_loss_advanced``.
    CROP_SIZE : int or None
        If set, training augmentation randomly crops each pair to a square
        of this side length (must be a multiple of ``PATCH_MULTIPLE``).
        ``None`` disables cropping. Applied only to the augmented (training)
        split — never at evaluation or inference.
    NLL_LOG_VAR_MIN : float
        Lower clip bound for the second output channel of every
        heteroscedastic model, for numerical stability. Reused as-is for
        two different distributional interpretations of that channel:
        Gaussian log-variance (``efficientnet_unet_nll``, until Round 2)
        and Laplace log-scale (``unet_nll``/``resunet_nll``/
        ``attention_unet_nll``, ``scripts.losses.laplace_nll_loss``) — the
        ``[-6, 6]`` range is a reasonable clip bound for either, so the
        field is shared rather than duplicated per distribution.
    NLL_LOG_VAR_MAX : float
        Upper clip bound, see above.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    IR_DIR: Path = _ROOT / "data" / "ir"
    RGB_DIR: Path = _ROOT / "data" / "rgb"
    MODELS_DIR: Path = _ROOT / "models"
    LOGS_DIR: Path = _ROOT / "logs"

    PATCH_MULTIPLE: int = 16
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 100
    SEED: int = 42

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15

    MOCKUP_ARTWORK_IDS: list[str] = [
        "tblu",
        "tbianco",
        "tbruno",
        "tgiallo",
        "trosso",
        "tverde",
    ]
    MOCKUP_TEST_RATIO: float = 0.05

    LOSS_ALPHA: float = 0.16
    CHARBONNIER_EPS: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    NLL_BETA: float = 0.5

    EARLY_STOPPING_PATIENCE: int = 20
    EARLY_STOPPING_MIN_DELTA: float = 0.0
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = False
    REDUCE_LR_FACTOR: float = 0.25
    REDUCE_LR_PATIENCE: int = 6
    REDUCE_LR_COOLDOWN: int = 2
    REDUCE_LR_MIN_DELTA: float = 1e-4
    REDUCE_LR_MIN_LR: float = 1e-6

    ADV_LOSS_ALPHA: float = 0.5
    ADV_LOSS_BETA: float = 0.3
    ADV_LOSS_GAMMA: float = 0.2
    LAPLACIAN_LEVELS: int = 4

    CROP_SIZE: int | None = None

    NLL_LOG_VAR_MIN: float = -6.0
    NLL_LOG_VAR_MAX: float = 6.0


settings = Settings()
