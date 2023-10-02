from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, LeakyReLU, Dropout, Dense, UpSampling2D, Input, Activation, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import non_neg
import tensorflow as tf


def get_encoder(model_name, input_shape, weights):
    """
    Get an encoder model based on the specified architecture.
    
    Args:
        model_name (str): Name of the architecture ('vgg16' or 'resnet50').
        input_shape (tuple): Shape of the input image (height, width, channels).
        weights (str or None): Weights to use ('imagenet' or None).

    Returns:
        tf.keras.Model: Encoder model.
    """
    # Load the model
    if model_name == "vgg16":
        model = VGG16(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "resnet50":
        model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError("Model name not recognized")
    # Add a global average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    encoder = tf.keras.Model(inputs=model.input, outputs=x)
    return encoder

def get_rank_decoder(encoder, initializations, num_neurons, num_layers):
    """
    Get a rank decoder model based on the specified encoder.

    Args:
        encoder (tf.keras.Model): Encoder model to build the decoder upon.
        initializations (str): Initialization method for dense layers.
        num_neurons (int): Number of neurons in the dense layers.
        num_layers (int): Number of dense layers.

    Returns:
        tf.keras.Model: Rank decoder model.
    """
    x_in = Input(encoder.output.shape[1:])
    x = x_in
    # Add dense layers
    for i in range(num_layers):
        x = Dense(num_neurons, activation='relu', kernel_initializer=initializations)(x)
    # Add a final dense layer
    x = Dense(1, activation=None, kernel_constraint = non_neg())(x)
    predictions = Activation('relu', dtype='float32')(x)
    return tf.keras.Model(inputs=x_in, outputs=predictions, name="rank_decoder")
