import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def build_resnet_colorizer(input_shape=(224, 224, 1)):
    """ResNet Encoder + Custom Decoder for Image Colorization."""
    inputs = layers.Input(shape=input_shape)
    
    # Convert 1 channel to 3 channels to use pretrained weights
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Feature Extractor (Encoder)
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    encoder_output = resnet.output # Shape is (7, 7, 2048) for 224x224 input
    
    # Decoder
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling2D((2, 2))(x) # 14x14
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 28x28
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 56x56
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 112x112
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 224x224
    
    outputs = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    model = Model(inputs, outputs, name='resnet_colorizer')
    return model
