from keras.models import load_model
import cv2
import numpy as np

recognizer = load_model('recognizer.h5')
img = cv2.imread('contours/85.png', cv2.IMREAD_GRAYSCALE)
img = img.reshape(1, 1, img.shape[0], img.shape[0])
prediction = recognizer.predict(img)[0]
print(np.argmax(prediction))
