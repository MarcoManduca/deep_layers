import logging
import tensorflow as tf
from pathlib import Path
from typing import Sequence as SequenceType

logger = logging.getLogger(__name__)

# --- HELPERS (to keep code clean) ---
def downsample(filters, size, apply_batchnorm=True, leaky_alpha=0.2):
    """Create a downsampling block with optional batch normalization.

    Args:
        filters: Number of output filters.
        size: Kernel size for Conv2D.
        apply_batchnorm: Whether to apply batch normalization. Defaults to True.
        leaky_alpha: Alpha (slope) for LeakyReLU activation. Defaults to 0.2.

    Returns:
        A tf.keras.Sequential model with Conv2D, optional BatchNormalization, and LeakyReLU.
    """
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=not apply_batchnorm))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    return result

def upsample(filters, size, apply_dropout=False, dropout_rate=0.5):
    """Create an upsampling block with optional dropout.

    Args:
        filters: Number of output filters.
        size: Kernel size for Conv2DTranspose.
        apply_dropout: Whether to apply dropout. Defaults to False.
        dropout_rate: Dropout rate if apply_dropout is True. Defaults to 0.5.

    Returns:
        A tf.keras.Sequential model with Conv2DTranspose, BatchNormalization, optional Dropout, and ReLU.
    """
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(dropout_rate))
    result.add(tf.keras.layers.ReLU())
    return result

# --- ARCHITECTURES ---

def build_generator(image_size=512, base_filters=64, **kwargs):
    """Build a U-Net style generator with skip connections.

    Args:
        image_size: Input image size (assumes square inputs). Defaults to 512.
        base_filters: Base number of filters; encoder/decoder scales this value. Defaults to 64.
        **kwargs: Additional kwargs for compatibility (ignored).

    Returns:
        A tf.keras.Model that takes RGB input and outputs 1-channel IR image.
    """
    # Use image_size to define input shape
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Encoder
    d1 = downsample(base_filters, 4, apply_batchnorm=True)(inputs)
    d2 = downsample(base_filters * 2, 4)(d1)
    d3 = downsample(base_filters * 4, 4)(d2)
    d4 = downsample(base_filters * 8, 4)(d3)
    d5 = downsample(base_filters * 16, 4)(d4)
    d6 = downsample(base_filters * 16, 4)(d5)
    d7 = downsample(base_filters * 16, 4)(d6)
    
    # Bottleneck
    b = downsample(base_filters * 16, 4)(d7)

    # Decoder with skip connections
    u1 = upsample(base_filters * 16, 4, apply_dropout=True)(b) 
    u1 = tf.keras.layers.Concatenate()([u1, d7])
    
    u2 = upsample(base_filters * 16, 4, apply_dropout=True)(u1)
    u2 = tf.keras.layers.Concatenate()([u2, d6])

    u3 = upsample(base_filters * 16, 4, apply_dropout=True)(u2)
    u3 = tf.keras.layers.Concatenate()([u3, d5])

    u4 = upsample(base_filters * 8, 4)(u3)
    u4 = tf.keras.layers.Concatenate()([u4, d4])

    u5 = upsample(base_filters * 4, 4)(u4)
    u5 = tf.keras.layers.Concatenate()([u5, d3])

    u6 = upsample(base_filters * 2, 4)(u5)
    u6 = tf.keras.layers.Concatenate()([u6, d2])

    u7 = upsample(base_filters, 4)(u6)
    u7 = tf.keras.layers.Concatenate()([u7, d1])

    last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')(u7)

    return tf.keras.Model(inputs=inputs, outputs=last, name="generator")

def build_discriminator(image_size=512, base_filters=64, **kwargs):
    """Build a PatchGAN discriminator with conditional input.

    Args:
        image_size: Input image size (assumes square inputs). Defaults to 512.
        base_filters: Base number of filters; scales with depth. Defaults to 64.
        **kwargs: Additional kwargs for compatibility (ignored).

    Returns:
        A tf.keras.Model that takes [IR image, RGB image] and outputs patch-wise predictions.
    """
    inp = tf.keras.layers.Input(shape=[image_size, image_size, 3], name='condition_input')
    tar = tf.keras.layers.Input(shape=[image_size, image_size, 1], name='image_input')

    x = tf.keras.layers.concatenate([tar, inp]) 

    # Encoder with more layers and progressive channel scaling
    x = downsample(base_filters, 4, apply_batchnorm=False)(x)
    x = downsample(base_filters * 2, 4)(x)
    x = downsample(base_filters * 4, 4)(x)
    x = downsample(base_filters * 8, 4)(x)
    x = downsample(base_filters * 16, 4)(x)
    x = downsample(base_filters * 16, 4)(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(base_filters * 16, 4, strides=1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    last = tf.keras.layers.Conv2D(1, 4, strides=1)(x)

    return tf.keras.Model(inputs=[tar, inp], outputs=last, name="discriminator")

# --- LOSS FUNCTIONS WITH LABEL SMOOTHING ---

def generator_loss(disc_generated_output, gen_output, target, l1_lambda=100.0):
    """Compute generator loss with GAN loss and L1 reconstruction loss.

    Args:
        disc_generated_output: Discriminator output for generated images.
        gen_output: Generated images from the generator.
        target: Target real images.
        l1_lambda: Weight for L1 loss. Defaults to 50.0 (reduced for stability).

    Returns:
        Weighted sum of GAN loss and L1 loss.
    """
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    return gan_loss + (l1_lambda * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output, label_smoothing=0.9):
    """Compute discriminator loss with label smoothing.

    Args:
        disc_real_output: Discriminator output for real images.
        disc_generated_output: Discriminator output for generated images.
        label_smoothing: Target value for real labels instead of 1.0. Defaults to 0.9.

    Returns:
        Sum of real and generated losses.
    """
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Label smoothing: use label_smoothing value instead of 1.0 for real images
    real_loss = loss_object(tf.ones_like(disc_real_output) * label_smoothing, disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    return real_loss + generated_loss


@tf.function
def train_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    rgb_images: tf.Tensor,
    real_images: tf.Tensor,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Run a single training step for the generator and discriminator.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        rgb_images: Batch of RGB inputs.
        real_images: Batch of real IR targets.
        gen_optimizer: Generator optimizer.
        disc_optimizer: Discriminator optimizer.

    Returns:
        Tuple of (generator_loss, discriminator_loss).
    """
    if len(real_images.shape) == 3:
        real_images = tf.expand_dims(real_images, axis=-1)

    with tf.GradientTape(persistent=True) as tape:
        generated_images = generator(rgb_images, training=True)

        disc_real_output = discriminator([real_images, rgb_images], training=True)
        disc_generated_output = discriminator([generated_images, rgb_images], training=True)

        gen_loss = generator_loss(disc_generated_output, generated_images, real_images, l1_lambda=50.0)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, label_smoothing=0.9)

    generator_gradients = tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_gan(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    dataset: SequenceType,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    epochs: int,
    checkpoint_dir: Path | None = None,
    log_every: int = 50,
) -> None:
    """Train the GAN for a number of epochs over a Sequence dataset with periodic checkpointing.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        dataset: A Sequence of (rgb_batch, ir_batch).
        gen_optimizer: Generator optimizer.
        disc_optimizer: Discriminator optimizer.
        epochs: Number of epochs.
        checkpoint_dir: Optional directory to save checkpoints every epoch.
        log_every: Print losses every N steps (and always on the last step).

    Raises:
        OSError: If checkpoint directory cannot be created.
    """
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs")
    num_steps = len(dataset)
    logger.info(f"Dataset size: {num_steps} batches")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        epoch_gen_losses = []
        epoch_disc_losses = []
        
        for step_idx in range(num_steps):
            try:
                rgb_images, ir_images = dataset[step_idx]
            except Exception as e:
                logger.error(f"Error loading batch {step_idx}: {e}")
                continue
            
            gen_loss, disc_loss = train_step(
                generator,
                discriminator,
                tf.convert_to_tensor(rgb_images, dtype=tf.float32),
                tf.convert_to_tensor(ir_images, dtype=tf.float32),
                gen_optimizer,
                disc_optimizer,
            )
            
            epoch_gen_losses.append(float(gen_loss))
            epoch_disc_losses.append(float(disc_loss))
            
            if (step_idx + 1) % log_every == 0 or (step_idx + 1) == num_steps:
                logger.info(
                    f"Step {step_idx + 1}/{num_steps}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}"
                )
        
        if not epoch_gen_losses:
            logger.warning(f"Epoch {epoch + 1} had no valid batches; skipping averages and checkpoints")
            continue

        # Log epoch averages
        avg_gen_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        avg_disc_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)
        logger.info(f"Epoch {epoch + 1} - Avg Gen Loss: {avg_gen_loss:.4f}, Avg Disc Loss: {avg_disc_loss:.4f}")
        
        # Save checkpoint every epoch
        if checkpoint_dir:
            gen_path = checkpoint_dir / f"generator_epoch_{epoch + 1}.keras"
            disc_path = checkpoint_dir / f"discriminator_epoch_{epoch + 1}.keras"
            generator.save(gen_path)
            discriminator.save(disc_path)
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    logger.info("Training completed")

