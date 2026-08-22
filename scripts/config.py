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
        Weight of the MAE term in the combined MAE + (1 - SSIM) loss.
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
        Lower clip bound for the predicted ``log_var`` channel in
        heteroscedastic (mu, log-variance) models, for numerical stability.
    NLL_LOG_VAR_MAX : float
        Upper clip bound for the predicted ``log_var`` channel.
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

    LOSS_ALPHA: float = 0.7

    ADV_LOSS_ALPHA: float = 0.5
    ADV_LOSS_BETA: float = 0.3
    ADV_LOSS_GAMMA: float = 0.2
    LAPLACIAN_LEVELS: int = 4

    CROP_SIZE: int | None = None

    NLL_LOG_VAR_MIN: float = -6.0
    NLL_LOG_VAR_MAX: float = 6.0


settings = Settings()
