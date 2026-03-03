
import os
from services.img import extract_path_list, RGBIRDataset
from services.gan import build_generator, build_discriminator, train_gan

# define environment variables
batch_size = int(os.getenv('BATCH_SIZE'))
epochs = int(os.getenv('EPOCHS'))
img_size = int(os.getenv('IMG_SIZE'))
kernel_size = int(os.getenv('KERNEL_SIZE'))
strides = int(os.getenv('STRIDES'))
rgb_path = os.getenv('TRAIN_RGB_PATH')
rgb_path = extract_path_list(rgb_path)
irr_path = os.getenv('TRAIN_IRR_PATH')
irr_path = extract_path_list(irr_path)


dataset = RGBIRDataset(rgb_path, irr_path, batch_size, img_size)
generator = build_generator(img_size, (img_size, img_size, 3), kernel_size, strides)
discriminator = build_discriminator(img_size, kernel_size, strides)
train_gan(generator, discriminator, dataset, epochs)
generator.save(f'../models/generator.keras')
discriminator.save(f'../models/discriminator.keras')
