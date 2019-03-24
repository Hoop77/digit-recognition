import utils
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import cv2

X_train, y_train, X_test, y_test = utils.load_mnist_data()
recognizer = load_model('recognizer.h5')

selected_digit = 6

X_sub = []
for i, X in enumerate(X_test):
    y = y_test[i]
    digit = np.argmax(y)
    if digit == selected_digit:
        X_sub.append(X)
X_sub = np.array(X_sub)

def plot_mnist():
    rows = 5
    cols = 5
    n = rows * cols
    offset = 2 * n
    fig, axes = plt.subplots(rows, cols)
    for i in range(n):
        X = X_sub[i + offset].reshape(1, 1, 28, 28)
        prediction = recognizer.predict(X)[0]
        digit = np.argmax(prediction)
        r = int(i / cols)
        c = int(i % cols)
        ax = axes[r][c]
        ax.imshow(X.reshape(28, 28), cmap=plt.get_cmap('gray'))
        ax.set_xlabel(str(digit))

    # show the plot
    plt.show()

def plot_handwritten():
    rows = 3
    cols = 3
    n = rows * cols
    offset = 0 * n
    fig, axes = plt.subplots(rows, cols)
    for i in range(n):
        X = cv2.imread('tmp/{}.png'.format(i), cv2.IMREAD_GRAYSCALE).reshape(1, 1, 28, 28)
        prediction = recognizer.predict(X)[0]
        digit = np.argmax(prediction)
        r = int(i / cols)
        c = int(i % cols)
        ax = axes[r][c]
        ax.imshow(X.reshape(28, 28), cmap=plt.get_cmap('gray'))
        ax.set_xlabel(str(digit))

    # show the plot
    plt.show()

plot_mnist()