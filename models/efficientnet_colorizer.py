import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def build_efficientnet_colorizer(input_shape=(None, None, 1)):
    """EfficientNet Encoder + Custom Decoder for Image Colorization."""
    inputs = layers.Input(shape=input_shape)
    
    # Convert 1 channel to 3 channels to use pretrained weights
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Feature Extractor (Encoder)
    efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    efficientnet.trainable = False  # Freeze pretrained EfficientNet encoder
    
    encoder_output = efficientnet.output 
    
    # Custom Decoder (with BatchNormalization and ReLU)
    x = layers.Conv2D(512, (3, 3), padding='same')(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = Model(inputs, outputs, name='efficientnet_colorizer')
    return model
