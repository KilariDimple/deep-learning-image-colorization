import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing import load_dataset, split_dataset, create_data_generator
from models.cnn_autoencoder import build_cnn_autoencoder
from models.resnet_colorizer import build_resnet_colorizer
from models.efficientnet_colorizer import build_efficientnet_colorizer
from metrics import psnr, ssim

def plot_history(history, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train MSE Loss')
    plt.plot(history.history['val_loss'], label='Val MSE Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_vs_epoch.png'))
    plt.close()

def main(args):
    # GPU Check
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPU Support Enabled: {physical_devices}")
    else:
        print("No GPU detected, falling back to CPU.")

    # Data Pipeline
    print(f"Loading dataset paths from {args.dataset_dir}...")
    file_paths = load_dataset(args.dataset_dir)
    if len(file_paths) == 0:
        print("No images found in the dataset directory.")
        return
        
    # Take a subset if specified
    if args.limit and args.limit < len(file_paths):
        file_paths = file_paths[:args.limit]
        
    train_paths, val_paths = split_dataset(file_paths, split_ratio=0.8)
    print(f"Training images: {len(train_paths)}, Validation images: {len(val_paths)}")
    
    train_ds = create_data_generator(train_paths, batch_size=args.batch_size, target_size=(224, 224), is_training=True)
    val_ds = create_data_generator(val_paths, batch_size=args.batch_size, target_size=(224, 224), is_training=False)
    
    # Model Selection
    print(f"Building {args.model} model...")
    if args.model == 'cnn':
        model = build_cnn_autoencoder()
    elif args.model == 'resnet':
        model = build_resnet_colorizer()
    elif args.model == 'efficientnet':
        model = build_efficientnet_colorizer()
    else:
        raise ValueError("Invalid model type")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='mse',
        metrics=[psnr, ssim]
    )
    
    # Callbacks
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{args.model}_best_model.h5")
    
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(patience=10, monitor='val_loss', mode='min', restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, monitor='val_loss', factor=0.5, min_lr=1e-6, verbose=1)
    ]
    
    # Training
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Plotting
    print("Saving training plots...")
    plot_history(history, save_dir=args.plot_dir)
    
    # Save final model as well
    final_model_path = os.path.join(args.save_dir, f"{args.model}_final_model.h5")
    model.save(final_model_path)
    print(f"Training complete. Models saved to {args.save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Image Colorizer")
    parser.add_argument('--dataset_dir', type=str, default='C:/Users/dimpl/Downloads/val2017', help='Path to dataset')
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'efficientnet'], default='efficientnet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for quick testing')
    parser.add_argument('--save_dir', type=str, default='models/saved_models')
    parser.add_argument('--plot_dir', type=str, default='plots')
    
    args = parser.parse_args()
    main(args)
