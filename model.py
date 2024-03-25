import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense
from tensorflow.keras.layers import Reshape, Lambda, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from utils import *
import os


def leaky_relu(x):
    """
    Leaky ReLU activation function

    Args:
        x (Tensor): Input tensor to apply leakey ReLU activation function

    Returns:
        Tensor with leakey ReLU activation function applied
    """
    return tf.nn.leaky_relu(x, alpha=0.2)


def initialize_model_params(img_height, img_width, img_channels, vocabulary_size):
    """
    Initialize and return the parameters for the CRNN model.

    Args:
        img_height (int): The height of the input images.
        img_width (int): The width of the input images.
        img_channels (int): The number of channels in the input images (e.g., 1 for grayscale or 3 for RGB).
        vocabulary_size (int): The size of the vocabulary, including all unique symbols/characters that the model should recognize.

    Returns:
        dict: A dictionary containing all the initialized parameters for the model.
    """
    path = os.getcwd()
    params = load_config(path + '/config/ctc_model.yaml')
    params.update({
        'img_height': img_height,
        'img_width': img_width,
        'img_channels': img_channels,
        'vocabulary_size': vocabulary_size
    })
    return params


def build_crnn_model(params):
    """
    Build and return the CRNN model based on the specified parameters.

    Args:
        params (dict): A dictionary containing parameters for the CRNN model. It should include keys for 'img_height','img_width', 'img_channels', 'conv_blocks', 'conv_filters', 'conv_kernel_size', 'conv_pooling_size','rnn_units', 'rnn_layers', and 'vocabulary_size'.

    Returns:
        Model: A TensorFlow Keras Model instance representing the compiled CRNN model.
    """
    # Define the input layer
    input_img = Input(shape=(
        params['img_height'], params['img_width'], params['img_channels']), name='image_input')

    x = input_img
    # Convolutional blocks
    for i in range(params['conv_blocks']):
        x = Conv2D(params['conv_filters'][i], params['conv_kernel_size']
                   [i], padding='same', activation=leaky_relu)(x)
        # Batch normalization layer
        x = BatchNormalization()(x)
        # Max pooling layer
        x = MaxPooling2D(pool_size=params['conv_pooling_size'][i])(x)

    # Prepare output for RNN layers
    new_shape = (-1, params['rnn_units'])
    x = Reshape(target_shape=new_shape)(x)
    # Dense layer to match the number of RNN units
    x = Dense(params['rnn_units'], activation=leaky_relu)(x)

    # Recurrent layers
    for _ in range(params['rnn_layers']):
        x = Bidirectional(
            LSTM(params['rnn_units'], return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = Dense(params['vocabulary_size'] + 1,
              activation='softmax', name='output')(x)

    # CTC loss layer
    labels = Input(name='labels', shape=[None,], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [labels, x, input_length, label_length])

    model = Model(inputs=[input_img, labels, input_length,
                  label_length], outputs=loss_out)
    return model


def ctc_lambda_func(args):
    """
    A lambda function to compute the CTC loss. This function is designed to be used within a Lambda layer in a Keras model, facilitating the calculation of the CTC loss between the predicted sequences and the true sequences.

    Args:
        args (list of Tensor): A list containing the following elements:
            y_true (Tensor): The true labels, represented as a sparse tensor.
            y_pred (Tensor): The logits from the model's output, before softmax activation.
            input_length (Tensor): The length of each input sequence, indicating how many timesteps each prediction spans.
            label_length (Tensor): The length of each ground truth sequence, indicating the number of true labels in each sequence.

    Returns:
        Tensor: A tensor representing the CTC loss for the batch. This tensor is typically used as the output of a Lambda layer, which can then be used to calculate the overall loss for training the model.
    """
    y_true, y_pred, input_length, label_length = args
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)
