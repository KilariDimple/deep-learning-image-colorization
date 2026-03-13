import os
import tensorflow as tf
import numpy as np
from skimage import color

def load_dataset(dataset_dir):
    """
    Returns a list of image file paths from the dataset directory.
    """
    supported_extensions = ('.jpg', '.jpeg', '.png')
    file_paths = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_paths.append(os.path.join(root, file))

    return file_paths


def preprocess_images(image_path, target_size=(224, 224)):
    """
    Reads an image, resizes it, converts RGB → LAB.
    Returns:
        L  : grayscale input
        ab : color channels
    """

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)

    # normalize RGB
    img = img / 255.0

    # convert RGB → LAB using skimage
    def rgb_to_lab(x):
        return color.rgb2lab(x)

    lab_img = tf.numpy_function(rgb_to_lab, [img], tf.float32)

    # IMPORTANT: restore shape information
    lab_img.set_shape([target_size[0], target_size[1], 3])

    # extract channels
    L = lab_img[..., 0:1] / 100.0
    ab = lab_img[..., 1:] / 128.0

    # set shapes explicitly
    L.set_shape([target_size[0], target_size[1], 1])
    ab.set_shape([target_size[0], target_size[1], 2])

    return L, ab


def create_data_generator(file_paths, batch_size=32, target_size=(224, 224), is_training=True):
    """
    Creates efficient tf.data pipeline.
    """

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    if is_training:
        dataset = dataset.shuffle(buffer_size=max(len(file_paths), 1000))

    def process_path(path):
        L, ab = preprocess_images(path, target_size=target_size)
        return L, ab

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def split_dataset(file_paths, split_ratio=0.8):
    """
    Splits dataset into train and validation.
    """
    np.random.shuffle(file_paths)
    split_idx = int(len(file_paths) * split_ratio)

    return file_paths[:split_idx], file_paths[split_idx:]