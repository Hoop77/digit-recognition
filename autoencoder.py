from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

num_samples = 100
x_train = x_train[:num_samples]
y_train = y_train[:num_samples]

def train_autoencoder():
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    tensorboard = TensorBoard(log_dir="tensorboard/autoencoder")
    autoencoder.fit(x_train, x_train,
                    epochs=1000,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[tensorboard])
    encoder.save('encoder.h5')
    autoencoder.save('autoencoder.h5')

def plot(autoencoder):
    decoded_imgs = autoencoder.predict(x_test)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def train_recognizer(encoder):
    input_img = Input(shape=(28, 28, 1))
    x = encoder(input_img)
    flattened = Flatten()
    x = flattened(x)
    num_encoder_neurons = np.prod(flattened.output_shape[1:])
    x = Dense(num_encoder_neurons, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    recognizer = Model(input_img, x)
    for i, layer in enumerate(encoder.layers):
        encoder.layers[i].trainable = False
    recognizer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    tensorboard = TensorBoard(log_dir="tensorboard/recognizer")
    recognizer.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=100, verbose=2, callbacks=[tensorboard])
    # Final evaluation of the model
    scores = recognizer.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: {}".format(100-scores[1]*100))

#train_autoencoder()
encoder = load_model('encoder.h5')
autoencoder = load_model('autoencoder.h5')
plot(autoencoder)
#train_recognizer(encoder)