# Plot ad hoc mnist instances
from keras.datasets import mnist
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import os
import cv2

IMG_SIZE = 28

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

def create_digit_labels(num_labels, digit):
    y = np.zeros(10)
    y[digit] = 1.0
    return np.array([y for i in range(num_labels)])

def load_custom_data(directory, digit, test_size=100):
    num_samples = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    if num_samples < test_size:
        raise ValueError('test_size too big!')
    X = np.empty([num_samples, IMG_SIZE, IMG_SIZE], dtype='float32')
    for i in range(num_samples):
        img = cv2.imread(os.path.join(directory, '{}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
        X[i] = img.astype('float32') / 255
    y_train = create_digit_labels(num_samples - test_size, digit)
    y_test = create_digit_labels(test_size, digit)
    np.random.shuffle(X)
    X = X.reshape(len(X), 1, IMG_SIZE, IMG_SIZE)
    X_train = X[test_size:]
    X_test = X[:test_size]
    return X_train, y_train, X_test, y_test    

def scale(img, factor):
    img = cv2.resize(img, None, fx=factor, fy=factor)
    offset = img.shape[0] - IMG_SIZE
    if offset > 0:
        img = img[offset : offset + IMG_SIZE, offset : offset + IMG_SIZE]
    elif offset < 0:
        offset = -offset
        border_size = int(offset / 2)
        extra = offset % 2
        img = cv2.copyMakeBorder(img, border_size + extra, border_size, border_size + extra, border_size, cv2.BORDER_CONSTANT)
    return img

def rotate(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))

def augument(img):
    angle = np.random.uniform(0, 20)
    img = rotate(img, angle)
    scale_factor = np.random.uniform(0.9, 1.1)
    img = scale(img, scale_factor)
    return img

def create_augumented_data(X_train, num_augumented_samples, digit):
    X_train_aug = np.empty([num_augumented_samples, 1, IMG_SIZE, IMG_SIZE])
    for i in range(num_augumented_samples):
        img = X_train[i % len(X_train), 0].copy()
        img = augument(img)
        X_train_aug[i, 0] = img
    y_train_aug = create_digit_labels(num_augumented_samples, digit)
    return X_train_aug, y_train_aug

def load_combined_data():
    num_ones_augumented_samples = 3000
    num_ones_mnist_samples = 3000

    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = load_mnist_data()
    X_train_custom, y_train_custom, X_test_custom, y_test_custom = load_custom_data('./datasets/1', 1)
    X_train_aug, y_train_aug = create_augumented_data(X_train_custom, num_ones_augumented_samples, 1)

    X_train = []
    y_train = []
    one_counter = 0
    for i, X in enumerate(X_train_mnist):
        y = y_train_mnist[i]
        digit = np.argmax(y)
        if digit != 1 or one_counter < num_ones_mnist_samples:
            X_train.append(X)
            y_train.append(y)
        if digit == 1:
            one_counter += 1
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = np.append(X_train, X_train_custom, axis=0)
    X_train = np.append(X_train, X_train_aug, axis=0)
    y_train = np.append(y_train, y_train_custom, axis=0)
    y_train = np.append(y_train, y_train_aug, axis=0)
       
    return X_train, y_train, \
           np.append(X_test_mnist, X_test_custom, axis=0), \
           np.append(y_test_mnist, y_test_custom, axis=0)