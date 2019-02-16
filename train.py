# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from time import time
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 3
np.random.seed(seed)

epochs = 10
batch_size = 128

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def plot():
    rows = 3
    cols = 3
    n = rows * cols
    offset = 2 * n
    fig, axes = plt.subplots(rows, cols)
    for i in range(n):
        r = int(i / cols)
        c = int(i % cols)
        axes[r][c].imshow(X_train[i + offset], cmap=plt.get_cmap('gray'))

    # show the plot
    plt.show()

# # flatten 28*28 images to a 784 vector for each image
# num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def multilayer_perceptron_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def simple_conv_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def larger_conv_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_conv_model()
num_samples = len(X_train)
# tensorboard for logging
tensorboard = TensorBoard(log_dir='tensorboard/recognizer_{}'.format(num_samples))
X_train_sub = X_train[:num_samples]
y_train_sub = y_train[:num_samples]
print('training set size: {}'.format(num_samples))
# Fit the model
model.fit(X_train_sub, y_train_sub, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Baseline Error: {}'.format(100-scores[1]*100))
model.save('recognizer.h5')