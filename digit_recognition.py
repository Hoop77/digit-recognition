import cv2
import numpy as np 
from keras.models import load_model
from matplotlib import pyplot as plt

recognizer = load_model('recognizer.h5')

MIN_WIDTH = 30
MIN_HEIGHT = 30

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png
orig = cv2.pyrDown(cv2.imread('img/1.jpg'))

k = 5
img = cv2.GaussianBlur(orig, (k, k), 0, 0)

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                110, 255, cv2.THRESH_BINARY_INV)

# find contours and get the external one
contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for i, c in enumerate(contours):
    # only use top-level contours
    if (hierarchy[0][i][3] != -1):
        continue
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        continue
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    fig = threshed_img[y:y+h, x:x+w]
    fig = cv2.resize(fig, (24, 24))
    fig = cv2.copyMakeBorder(fig, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
    cv2.imwrite("tmp/{}.png".format(i), fig)
    fig = fig.reshape(1, 1, fig.shape[0], fig.shape[0])
    prediction = recognizer.predict(fig)[0]
    num = np.argmax(prediction)
    cv2.putText(img, str(num), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=3)

    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    cv2.drawContours(img, [box], 0, (0, 0, 255))

cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()