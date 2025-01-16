
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Input, Concatenate

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)


def build_generator(image_size = 128, input_shape = (128, 128, 3), kernel_size = 5, strides = 2) -> tf.keras.Model:
    """
    Build the generator model.
    Args:
        input_shape: shape of the input tensor (height, width, channels).
    Returns:
        tf.keras.Model: the generator model.
    """
    inputs = Input(shape = input_shape)

    # encoder
    down1 = Conv2D(int(image_size/128), kernel_size = kernel_size, strides = strides, padding = "same")(inputs)
    down1 = LeakyReLU(negative_slope = 0.2)(down1)
    down2 = Conv2D(int(image_size/64), kernel_size = kernel_size, strides = strides, padding = "same")(down1)
    down2 = LeakyReLU(negative_slope = 0.2)(down2)
    down3 = Conv2D(int(image_size/32), kernel_size = kernel_size, strides = strides, padding = "same")(down2)
    down3 = LeakyReLU(negative_slope = 0.2)(down3)
    down4 = Conv2D(int(image_size/16), kernel_size = kernel_size, strides = strides, padding = "same")(down3)
    down4 = LeakyReLU(negative_slope = 0.2)(down4)
    down5 = Conv2D(int(image_size/8), kernel_size = kernel_size, strides = strides, padding = "same")(down4)
    down5 = LeakyReLU(negative_slope = 0.2)(down5)
    down6 = Conv2D(int(image_size/4), kernel_size = kernel_size, strides = strides, padding = "same")(down5)
    down6 = LeakyReLU(negative_slope = 0.2)(down6)
    down7 = Conv2D(int(image_size/2), kernel_size = kernel_size, strides = strides, padding = "same")(down6)
    down7 = LeakyReLU(negative_slope = 0.2)(down7)

    # bottleneck
    bottleneck = Conv2D(image_size, kernel_size = kernel_size, strides = strides, padding = "same")(down7)
    bottleneck = LeakyReLU(negative_slope = 0.2)(bottleneck)

    # decoder
    up1 = Conv2DTranspose(int(image_size/2), kernel_size = kernel_size, strides = strides, padding = "same")(bottleneck)
    up1 = Dropout(0.5)(up1)
    up1 = Concatenate()([up1, down7])
    up2 = Conv2DTranspose(int(image_size/4), kernel_size = kernel_size, strides = strides, padding = "same")(up1)
    up2 = Dropout(0.5)(up2)
    up3 = Conv2DTranspose(int(image_size/8), kernel_size = kernel_size, strides = strides, padding = "same")(up2)
    up3 = Dropout(0.5)(up3)
    up4 = Conv2DTranspose(int(image_size/16), kernel_size = kernel_size, strides = strides, padding = "same")(up3)
    up4 = Dropout(0.5)(up4)
    up4 = Concatenate()([up4, down4])
    up5 = Conv2DTranspose(int(image_size/32), kernel_size = kernel_size, strides = strides, padding = "same")(up4)
    up5 = Dropout(0.5)(up5)
    up6 = Conv2DTranspose(int(image_size/64), kernel_size = kernel_size, strides = strides, padding = "same")(up5)
    up6 = Dropout(0.5)(up6)
    up7 = Conv2DTranspose(int(image_size/128), kernel_size = kernel_size, strides = strides, padding = "same")(up6)
    up7 = Dropout(0.5)(up7)
    up7 = Concatenate()([up7, down1])

    outputs = Conv2DTranspose(1, kernel_size = kernel_size, strides = strides, activation = 'tanh', padding = "same")(up7)

    return tf.keras.Model(inputs, outputs, name = "generator")


def build_discriminator(image_size = 128, kernel_size = 5, strides = 2) -> tf.keras.Model:
    """
    Build the contitional discriminator model.
    Returns:
        Un modello Keras che accetta due input: l'immagine (reale o generata) e l'immagine RGB.
    """
    # input image IR
    image_input = tf.keras.layers.Input(shape=(image_size, image_size, 1), name="image_input")

    # input condition RGB
    condition_input = tf.keras.layers.Input(shape=(image_size, image_size, 3), name="condition_input")

    # concat inputs along the channel axis
    combined_input = tf.keras.layers.Concatenate()([image_input, condition_input])

    # convolutional network for the discriminator
    x = tf.keras.layers.Conv2D(int(image_size/128), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(combined_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/64), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/32), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/16), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/8), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/4), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(int(image_size/2), kernel_size = (kernel_size, kernel_size), strides = (strides, strides), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    # discriminator model
    discriminator = tf.keras.Model(inputs=[image_input, condition_input], outputs=x, name="discriminator")
    return discriminator


def generator_loss(disc_generated_output, gen_output, target):
    """
    Calculate the generator loss by combining adversarial loss and L1 loss.
    Args:
        disc_generated_output: Discriminator output for the generated image.
        gen_output: Generated image from the generator.
        target: Real target image (IR).
        no_drawing: Flag indicating if the target image is devoid of underlying drawings.
    Returns:
        The total generator loss.
    """
    # expand target dimensions if needed
    if len(target.shape) == 3:  # from [batch_size, height, width]
        target = tf.expand_dims(target, axis=-1)  # to [batch_size, height, width, 1]
    
    # L1 loss calculation (similarity between generated image and target)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # calculate adversarial loss
    adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output
    )

    # combine adversarial loss with L1 loss
    total_gen_loss = adversarial_loss + (100 * l1_loss)  # weight the L1 loss
    return total_gen_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Calculate the discriminator loss.
    Args:
        disc_real_output: Discriminator output for the real image.
        disc_generated_output: Discriminator output for the generated image.
    Returns:
        The total discriminator loss.
    """
    # real image loss
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output
    )
    # generated image loss
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    # total discriminator loss
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


@tf.function
def train_step(generator, discriminator, rgb_images, real_images, gen_optimizer, disc_optimizer):
    """
    Single training step for the generator and discriminator.
    Args:
        generator: generator model.
        discriminator: discriminator model.
        rgb_images: RGB images.
        real_images: IR images.
        gen_optimizer: Optimizer for the generator.
        disc_optimizer: Optimizer for the discriminator.
    Returns:
        Tuple of generator loss and discriminator loss.
    """
    # add missing channel for target if needed
    if len(real_images.shape) == 3:  # from [batch_size, height, width]
        real_images = tf.expand_dims(real_images, axis=-1)  # to [batch_size, height, width, 1]

    with tf.GradientTape(persistent=True) as tape:
        # generate image from generator
        generated_images = generator(rgb_images, training = True)

        # obtain discriminator output
        # pass both real and RGB image as input
        disc_real_output = discriminator([real_images, rgb_images], training=True)
        disc_generated_output = discriminator([generated_images, rgb_images], training=True)

        # loss calculation
        gen_loss = generator_loss(disc_generated_output, generated_images, real_images)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # gradients calculation
    generator_gradients = tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    # update weights
    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_gan(generator, discriminator, dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for img in range(len(dataset)):
            rgb_images, ir_images = dataset[img]
            gen_loss, disc_loss = train_step(generator, discriminator,
                                             tf.convert_to_tensor(rgb_images, dtype = tf.float32),
                                             tf.convert_to_tensor(ir_images, dtype = tf.float32),
                                             generator_optimizer,
                                             discriminator_optimizer)
            print(f"Step {img+1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
