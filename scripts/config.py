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
    LOSS_ALPHA : float
        Weight of the MAE term in the combined MAE + (1 - SSIM) loss.
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

    LOSS_ALPHA: float = 0.7


settings = Settings()
