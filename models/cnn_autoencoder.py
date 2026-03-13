import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn_autoencoder(input_shape=(224, 224, 1)):
    """Simple CNN Autoencoder for image colorization."""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    
    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer: 2 channels for a and b
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = Model(inputs, outputs, name='cnn_autoencoder')
    return model
