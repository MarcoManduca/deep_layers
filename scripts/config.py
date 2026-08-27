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
    KFOLD_K : int
        Number of folds for Round 4 grouped cross-validation
        (``scripts.kfold``, ``fixing.md`` #4). Real artworks are partitioned
        into this many groups by artwork ID; mockups always stay in train.
    KFOLD_SEED : int
        Seed for the fold assignment, kept separate from ``SEED`` so the
        k-fold partition is stable even if the global training seed changes.
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
    GRADIENT_CLIP_VALUE : float
        Per-element gradient clip passed to ``Adam(clipvalue=...)``.
        Guards against ``tf.image.ssim_multiscale``'s known gradient
        singularity: its per-scale power terms use fractional exponents
        (the standard MS-SSIM power factors), and differentiating ``x**p``
        at ``x=0`` (a scale's structural term collapsing to exactly zero,
        clamped there internally by TF) gives ``0**(p-1)`` with ``p-1<0``,
        i.e. ``+Inf``. Unlike ``clipnorm`` — which can turn an already-Inf
        gradient into ``NaN`` via ``Inf/Inf`` when computing the global
        norm — per-element ``clipvalue`` caps each entry directly, turning
        ``Inf`` into a large finite number instead (`fixing.md` §7).
    NLL_BETA : float
        Weighting exponent for the beta-weighted NLL losses
        (``scripts.losses.beta_gaussian_nll_loss``,
        ``scripts.losses.laplace_nll_loss``).
    EARLY_STOPPING_PATIENCE : int
        Epochs with no ``val_loss`` improvement before training stops.
    EARLY_STOPPING_MIN_DELTA : float
        Minimum ``val_loss`` change to count as an improvement for
        early stopping. ``5e-4`` (not ``0.0``): observed on `unet_v2`'s
        Round 1/3 run (``fixing.md`` §7.1) — after `ReduceLROnPlateau`'s
        one LR drop, both train and val loss went flat (~0.003 total
        change over 14 epochs) while `0.0` let noise-level, sub-1e-3
        wiggles in the ~21-batch validation set keep resetting the
        patience counter, so training ran to the full `EPOCHS` cap well
        past the point of real improvement. `ModelCheckpoint` still saves
        the true best regardless of this value — raising it only affects
        when *training stops*, not what gets saved.
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
    DILATION_RATES : list[int]
        Dilation rate for each parallel branch of the Round 3 dilated
        bottleneck (``scripts.aspp.dilated_bottleneck``, ``fixing.md`` #7),
        used by ``unet_dilated``/``unet_v2_dilated``.
    CROP_SIZE : int or None
        If set, training augmentation randomly crops each pair to a square
        of this side length (must be a multiple of ``PATCH_MULTIPLE``).
        ``None`` disables cropping. Applied only to the augmented (training)
        split — never at evaluation or inference.
    NLL_LOG_VAR_MIN : float
        Lower clip bound for the second output channel of every
        heteroscedastic model, for numerical stability. Named for its
        original Gaussian log-variance interpretation, but since
        ``fixing.md`` #10 (Round 2) every NLL architecture — including
        ``efficientnet_unet_nll`` — is trained as a Laplace log-scale
        (``scripts.losses.laplace_nll_loss``) instead; the field is kept
        under its original name rather than renamed to ``NLL_LOG_SCALE_MIN``
        to avoid an unrelated rename churning every call site, and the
        ``[-6, 6]`` range remains a reasonable clip bound for a log-scale
        channel (it was never Gaussian-specific numerically, just in name).
    NLL_LOG_VAR_MAX : float
        Upper clip bound, see above.
    FINETUNE_LEARNING_RATE : float
        Adam learning rate for phase 2 of EfficientNet's two-phase
        fine-tuning (``fixing.md`` #6, Round 2): once the decoder has
        converged against the frozen pretrained encoder (phase 1, plain
        ``LEARNING_RATE``), phase 2 unfreezes the encoder
        (``freeze_encoder=False``) and continues training the whole
        network end-to-end. Conventionally 10-100x smaller than the
        from-scratch rate to avoid destroying the pretrained ImageNet
        features with large gradient steps early in phase 2; ``1e-5``
        (10x below ``LEARNING_RATE``) is a conservative choice within that
        range, erring toward preserving the pretrained weights rather than
        risking catastrophic forgetting.
    FINETUNE_EPOCHS : int
        Maximum epochs for phase 2. Phase 2 starts from an already
        converged decoder, not from scratch, so it needs a much shorter
        budget than ``EPOCHS`` — it is a refinement pass over already-good
        weights, not a full training run. Early stopping
        (``EARLY_STOPPING_PATIENCE``) still governs the actual stopping
        point; this is only the upper bound.
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

    KFOLD_K: int = 3
    KFOLD_SEED: int = 42

    LOSS_ALPHA: float = 0.16
    CHARBONNIER_EPS: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP_VALUE: float = 1.0
    NLL_BETA: float = 0.5

    EARLY_STOPPING_PATIENCE: int = 20
    EARLY_STOPPING_MIN_DELTA: float = 5e-4
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
    DILATION_RATES: list[int] = [1, 2, 4, 8]

    CROP_SIZE: int | None = None

    NLL_LOG_VAR_MIN: float = -6.0
    NLL_LOG_VAR_MAX: float = 6.0

    FINETUNE_LEARNING_RATE: float = 1e-5
    FINETUNE_EPOCHS: int = 20


settings = Settings()
