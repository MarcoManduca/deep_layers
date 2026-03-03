import logging

from deep_layers.config import TrainingConfig
from deep_layers.data import RGBIRDataset, extract_path_list
from deep_layers.trainer import GANTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train a GAN model for RGB to IR image generation.

    Raises:
        Exception: If training fails at any stage.
    """
    try:
        logger.info("Initializing configuration...")
        config = TrainingConfig()
        
        logger.info("Creating trainer instance...")
        trainer = GANTrainer(config)

        logger.info("Extracting image paths...")
        rgb_paths = extract_path_list(config.train_rgb_dir)
        ir_paths = extract_path_list(config.train_irr_dir)
        
        logger.info(f"Found {len(rgb_paths)} RGB images and {len(ir_paths)} IR images")

        logger.info("Creating dataset...")
        dataset = RGBIRDataset(
            rgb_paths=rgb_paths,
            ir_paths=ir_paths,
            batch_size=config.batch_size,
            target_size=config.img_size,
        )

        logger.info("Building models...")
        trainer.build_models()
        
        logger.info("Compiling optimizers...")
        trainer.compile_optimizers(learning_rate=2e-4, beta_1=0.5)
        
        logger.info("Starting training...")
        trainer.train(dataset, log_every=50)
        
        logger.info("Saving trained models...")
        trainer.save_models()
        
        logger.info("✓ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

