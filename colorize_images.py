import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import color

from models.cnn_autoencoder import build_cnn_autoencoder
from models.resnet_colorizer import build_resnet_colorizer
from models.efficientnet_colorizer import build_efficientnet_colorizer


def process_and_colorize(image_path, model, output_path, target_size=(224, 224)):
    """Reads image, colorizes it, and saves output."""

    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    # Resize for model
    img_resized = img.resize(target_size)
    img_np = np.array(img_resized) / 255.0

    # Convert RGB → LAB
    lab_img = color.rgb2lab(img_np)

    # Extract L channel
    L = lab_img[:, :, 0] / 100.0
    L = L.reshape(1, target_size[0], target_size[1], 1)

    # Predict AB channels
    pred_ab = model.predict(L)[0]
    pred_ab = pred_ab * 128.0

    # Reconstruct LAB image
    lab_output = np.zeros((target_size[0], target_size[1], 3))
    lab_output[:, :, 0] = lab_img[:, :, 0]
    lab_output[:, :, 1:] = pred_ab

    # Convert LAB → RGB
    rgb_output = color.lab2rgb(lab_output)

    # Resize back to original size
    rgb_output = Image.fromarray((rgb_output * 255).astype(np.uint8))
    rgb_output = rgb_output.resize(original_size)

    # Save image
    rgb_output.save(output_path)
    print(f"Saved: {output_path}")


def main(args):

    # Build model
    if args.model == 'cnn':
        model = build_cnn_autoencoder()
    elif args.model == 'resnet':
        model = build_resnet_colorizer()
    elif args.model == 'efficientnet':
        model = build_efficientnet_colorizer()
    else:
        raise ValueError("Invalid model type")

    # Load weights
    if os.path.exists(args.weights):
        model(tf.zeros((1, 224, 224, 1)))
        model.load_weights(args.weights)
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: weights file not found")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process input
    if os.path.isfile(args.input):
        output_path = os.path.join(args.output_dir, "colorized_" + os.path.basename(args.input))
        process_and_colorize(args.input, model, output_path)

    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(args.input, file)
                output_path = os.path.join(args.output_dir, "colorized_" + file)
                process_and_colorize(input_path, model, output_path)

    else:
        print("Invalid input path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize grayscale images")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to image or folder"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Folder to save colorized images"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet", "efficientnet"],
        default="efficientnet"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/saved_models/efficientnet_best_model.h5"
    )

    args = parser.parse_args()

    main(args)