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
    ADV_LOSS_ALPHA : float
        Weight of the MAE term in ``combined_loss_advanced``.
    ADV_LOSS_BETA : float
        Weight of the Laplacian pyramid term in ``combined_loss_advanced``.
    ADV_LOSS_GAMMA : float
        Weight of the FFT magnitude term in ``combined_loss_advanced``.
    LAPLACIAN_LEVELS : int
        Number of Laplacian pyramid levels in ``combined_loss_advanced``.
    NORM_LOSS_ALPHA : float
        Weight of the MAE term in ``combined_loss_normalized``.
    NORM_LOSS_BETA : float
        Weight of the per-window z-score term in ``combined_loss_normalized``.
        Not on the same scale as ``NORM_LOSS_ALPHA``: the normalised term runs
        roughly 9x larger than MAE on real predictions.
    ZSCORE_WINDOW : int
        Side length of the Gaussian window used by ``local_zscore_loss``.
    ZSCORE_SIGMA : float
        Standard deviation of that Gaussian window.
    ZSCORE_SIGMA_FLOOR : float
        Lower bound on the local standard deviation used as the z-score
        denominator; bounds the gradient in flat regions. Default is the 10th
        percentile of the local standard deviation measured on the IR set.
    ZSCORE_CLIP : float or None
        Per-pixel clip on the normalised error, or ``None`` to disable.
    CROP_SIZE : int or None
        If set, training augmentation randomly crops each pair to a square
        of this side length (must be a multiple of ``PATCH_MULTIPLE``).
        ``None`` disables cropping. Applied only to the augmented (training)
        split — never at evaluation or inference.
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

    ADV_LOSS_ALPHA: float = 0.5
    ADV_LOSS_BETA: float = 0.3
    ADV_LOSS_GAMMA: float = 0.2
    LAPLACIAN_LEVELS: int = 4

    NORM_LOSS_ALPHA: float = 1.0
    NORM_LOSS_BETA: float = 0.03
    ZSCORE_WINDOW: int = 11
    ZSCORE_SIGMA: float = 1.5
    ZSCORE_SIGMA_FLOOR: float = 0.01
    ZSCORE_CLIP: float | None = 4.0

    CROP_SIZE: int | None = None


settings = Settings()
