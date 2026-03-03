
import os
import cv2
import numpy as np
import tensorflow as tf

def load_images(rgb_path, ir_path) -> tuple :
    """Load RGB and IR images."""
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)  # IR is single-channel
    # check if images are valid
    if rgb_img is None or ir_img is None:
        raise ValueError(f"Image is not valid. RGB: {rgb_path}, IR: {ir_path}")
    return rgb_img, ir_img

def add_padding(image, target_size):
    if image is None:
        raise ValueError("Null image, check image validity")
    old_size = image.shape[:2]  # (height, width)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # (weight, width)

    # resize image
    resized_image = cv2.resize(image, new_size)

    # calculate padding
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # add padding
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def preprocess_images_with_padding(rgb_img, ir_img, target_size = 128) -> tuple:
    """
    Reduce images while preserving aspect ratio and add padding
    """
    # add padding to RGB & IR images
    rgb_img_padded = add_padding(rgb_img, (target_size, target_size))
    ir_img_padded = add_padding(ir_img, (target_size, target_size))

    # image normalization (0-1)
    rgb_img_padded = rgb_img_padded / float(target_size - 1)
    ir_img_padded = ir_img_padded / float(target_size - 1)

    return rgb_img_padded, ir_img_padded

class RGBIRDataset(tf.keras.utils.Sequence):
    def __init__(self, rgb_paths:list, ir_paths:list , batch_size:int = 1, target_size:tuple = 128):
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.rgb_paths) / self.batch_size))

    def __getitem__(self, idx):
        if idx < self.__len__() :
            batch_rgb_paths = self.rgb_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_ir_paths = self.ir_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_rgb, batch_ir = [], []
            for rgb_path, ir_path in zip(batch_rgb_paths, batch_ir_paths):
                if rgb_path.endswith('.jpg') and ir_path.endswith('.jpg'):
                    rgb_img, ir_img = load_images(rgb_path, ir_path)
                    rgb_img, ir_img = preprocess_images_with_padding(rgb_img, ir_img, self.target_size)
                    batch_rgb.append(rgb_img)
                    batch_ir.append(ir_img)

            return np.array(batch_rgb), np.array(batch_ir)

def extract_path_list(path:str) -> list:
    outlist = list()
    for _file in os.listdir(path):
        outlist.append(os.path.join(path, _file))
    return outlist

def generate_underpainting(generator, rgb_image):
    rgb_image = preprocess_images_with_padding(rgb_image, rgb_image, 512)[0]  # normalization (RGB only)
    rgb_image = np.expand_dims(rgb_image, axis = 0)  # add batch dimension
    generated_image = generator.predict(rgb_image)[0]
    return generated_image

def normalize_irr(image) -> np.ndarray:
    lower_percentile = np.percentile(image, 0)
    upper_percentile = np.percentile(image, 100)
    img_clipped = np.clip(image, lower_percentile, upper_percentile)
    img_normalized = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized

def remove_padding(original_image, padded_image):
    if original_image is None:
        raise ValueError("No original image, check image validity")
    if padded_image is None:
        raise ValueError("No padded image, check image validity")
    target_size = padded_image.shape
    old_size = original_image.shape[:2]  # (height, width)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # (height, width)

    # calculate padding
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # add padding
    if len(padded_image) == 3:
        depadded_image = padded_image[top:(target_size[0]-bottom), left:(target_size[1]-right), :]
    else:
        depadded_image = padded_image[top:(target_size[0]-bottom), left:(target_size[1]-right)]
    return depadded_image
