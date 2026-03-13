import os
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import color

from models.efficientnet_colorizer import build_efficientnet_colorizer

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 40

DATASETS = [
    ("datasets/data/Cars/cars_grey", "datasets/data/Cars/cars_colour"),
    ("datasets/data/Flowers/flowers_grey", "datasets/data/Flowers/flowers_colour")
]

def load_pair(gray_path, color_path):
    gray = Image.open(gray_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    color_img = Image.open(color_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    gray = np.array(gray) / 255.0
    color_img = np.array(color_img) / 255.0

    lab = color.rgb2lab(color_img)

    L = lab[:, :, 0] / 100.0
    ab = lab[:, :, 1:] / 128.0

    L = L.reshape(IMG_SIZE, IMG_SIZE, 1)

    return L.astype(np.float32), ab.astype(np.float32)

def augment_image(L, ab):
    # Combine to apply same spatial transformations
    combined = tf.concat([L, ab], axis=-1)
    
    # Random horizontal flip
    combined = tf.image.random_flip_left_right(combined)
    
    L_aug = combined[:, :, 0:1]
    ab_aug = combined[:, :, 1:]
    
    # Small brightness variation to L channel
    L_aug = tf.image.random_brightness(L_aug, max_delta=0.05)
    L_aug = tf.clip_by_value(L_aug, 0.0, 1.0)
    
    return L_aug, ab_aug

def iter_dataset_pairs():
    """Generator to yield image pairs lazily to avoid out-of-memory."""
    for gray_dir, color_dir in DATASETS:
        if not os.path.isdir(gray_dir) or not os.path.isdir(color_dir):
            continue
            
        gray_files = sorted(os.listdir(gray_dir))
        # Ensure pairing by exact same filename
        for g_file in gray_files:
            gray_path = os.path.join(gray_dir, g_file)
            color_path = os.path.join(color_dir, g_file)
            if os.path.exists(color_path):
                yield gray_path, color_path

def process_path(gray_path, color_path):
    def _tf_load(g_path, c_path):
        return load_pair(g_path.decode('utf-8'), c_path.decode('utf-8'))
        
    L, ab = tf.numpy_function(_tf_load, [gray_path, color_path], [tf.float32, tf.float32])
    L.set_shape([IMG_SIZE, IMG_SIZE, 1])
    ab.set_shape([IMG_SIZE, IMG_SIZE, 2])
    return L, ab

def create_dataset():
    pairs = list(iter_dataset_pairs())
    if not pairs:
        print("Warning: No dataset pairs found!")
        return tf.data.Dataset.from_tensor_slices(([], []))

    gray_paths, color_paths = zip(*pairs)
    
    dataset = tf.data.Dataset.from_tensor_slices((list(gray_paths), list(color_paths)))
    dataset = dataset.shuffle(max(len(pairs), 1000))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    print("Loading datasets with tf.data pipeline...")
    dataset = create_dataset()

    print(f"Building model ({IMG_SIZE}x{IMG_SIZE} resolution)...")
    model = build_efficientnet_colorizer((IMG_SIZE, IMG_SIZE, 1))

    # Output expects shape
    model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 1)))

    base_model_path = "models/saved_models/efficientnet_best_model.h5"
    if os.path.exists(base_model_path):
        print(f"Loading pretrained weights from {base_model_path}...")
        try:
            # Using by_name=True to safely manage newly added BatchNorm layers
            model.load_weights(base_model_path, by_name=True, skip_mismatch=True)
            print("Successfully loaded compatible base weights.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(f"Warning: {base_model_path} not found. Starting from scratch.")

    print("Fine-tuning model (Adam optimizer, Huber loss, 1e-5 LR)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.Huber()
    )

    try:
        next(iter(dataset))
        has_data = True
    except (StopIteration, tf.errors.InvalidArgumentError):
        has_data = False

    if has_data:
        model.fit(dataset, epochs=EPOCHS)
    else:
        print("Dataset was empty, skipping training loop.")

    print("Saving fine-tuned model...")
    os.makedirs("models/saved_models", exist_ok=True)
    model.save("models/saved_models/efficientnet_finetuned.h5")
    print("Completed!")

if __name__ == "__main__":
    main()