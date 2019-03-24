# Plot ad hoc mnist instances
from keras.datasets import mnist
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def load_mnist_data(flattened=False):
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if flattened:
        # flatten 28*28 images to a 784 vector for each image
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    else:
        # reshape to be [samples][pixels][width][height]
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test