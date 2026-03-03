import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from deep_layers.config import TrainingConfig, ensure_directories
from deep_layers.models import build_discriminator, build_generator, train_gan

logger = logging.getLogger(__name__)


class GANTrainer:
    """Encapsulates GAN training workflow."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration.

        Raises:
            FileNotFoundError: If required paths are missing.
            ValueError: If configuration values are invalid.
        """
        self.config = config
        
        # Validate configuration early
        try:
            config.validate()
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        ensure_directories(config)
        self._configure_devices()
        self._maybe_set_seed(config.seed)

        self.generator: tf.keras.Model | None = None
        self.discriminator: tf.keras.Model | None = None
        self.gen_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.disc_optimizer: tf.keras.optimizers.Optimizer | None = None
        
        logger.info("GANTrainer initialized successfully")

    @staticmethod
    def _maybe_set_seed(seed: int | None) -> None:
        """Optionally set random seeds for reproducibility.

        Args:
            seed: Seed value to set. If None, no seed is applied.
        """
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to {seed}")

    @staticmethod
    def _configure_devices() -> None:
        """Enable GPU usage if available (and set memory growth)."""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s); enabling memory growth")
            except RuntimeError as exc:
                logger.warning(f"Could not set GPU memory growth: {exc}")
        else:
            logger.info("No GPU detected; running on CPU")

    def build_models(self) -> None:
        """Build generator and discriminator models."""
        logger.info("Building models...")
        
        self.generator = build_generator(
            image_size=self.config.img_size,
            input_shape=(self.config.img_size, self.config.img_size, 3),
            kernel_size=self.config.kernel_size,
            strides=self.config.strides,
        )
        self.discriminator = build_discriminator(
            image_size=self.config.img_size,
            kernel_size=self.config.kernel_size,
            strides=self.config.strides,
        )
        logger.info(f"Generator: {self.generator.count_params():,} parameters")
        logger.info(f"Discriminator: {self.discriminator.count_params():,} parameters")

    def compile_optimizers(self, learning_rate: float = 2e-4, beta_1: float = 0.5) -> None:
        """Initialize optimizers for both models.

        Args:
            learning_rate: Optimizer learning rate.
            beta_1: Adam beta_1 parameter.
        """
        logger.info(f"Compiling optimizers with learning_rate={learning_rate}, beta_1={beta_1}")

        if tf.config.list_physical_devices("GPU"):
            self.gen_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=beta_1)
            self.disc_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=beta_1)
            logger.info(f"Using legacy Adam optimizer on Metal GPU")
        else:
            self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1)
            self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1)
            logger.info(f"Using standard Adam optimizer on CPU")

    def train(self, dataset, log_every: int = 50) -> None:
        """Train the GAN.

        Args:
            dataset: A Sequence of (rgb_batch, ir_batch).
            log_every: Print losses every N steps (and always on the last step).
        
        Raises:
            RuntimeError: If models or optimizers are not initialized.
        """
        if self.generator is None or self.discriminator is None:
            raise RuntimeError("Models not built. Call build_models() first.")
        if self.gen_optimizer is None or self.disc_optimizer is None:
            raise RuntimeError("Optimizers not compiled. Call compile_optimizers() first.")

        logger.info("Starting training phase...")
        
        train_gan(
            self.generator,
            self.discriminator,
            dataset,
            self.gen_optimizer,
            self.disc_optimizer,
            self.config.epochs,
            checkpoint_dir=self.config.checkpoint_dir,
            log_every=log_every,
        )

    def save_models(self) -> None:
        """Save trained models to disk.

        Raises:
            RuntimeError: If models are not trained.
        """
        if self.generator is None or self.discriminator is None:
            raise RuntimeError("Models not trained. Call train() first.")

        models_dir = Path(self.config.models_dir)
        gen_path = models_dir / "generator.keras"
        disc_path = models_dir / "discriminator.keras"
        
        self.generator.save(gen_path)
        self.discriminator.save(disc_path)
        
        logger.info(f"Models saved to {models_dir}")
        logger.info(f"  - Generator: {gen_path}")
        logger.info(f"  - Discriminator: {disc_path}")
