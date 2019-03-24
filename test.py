from keras.models import load_model
import cv2
import numpy as np
import utils
import os
from matplotlib import pyplot as plt

# recognizer = load_model('recognizer.h5')
# img = cv2.imread('tmp/5.png', cv2.IMREAD_GRAYSCALE)
# img = img.reshape(1, 1, img.shape[0], img.shape[0])
# prediction = recognizer.predict(img)[0]
# print(np.argmax(prediction))

X_train, y_train, _, _ = utils.load_combined_data()
print(X_train.shape[0])
print(y_train.shape[0])
