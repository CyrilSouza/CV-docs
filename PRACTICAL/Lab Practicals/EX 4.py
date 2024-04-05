#EXPERIMENT 4 :

import cv2 as cv

from matplotlib import pyplot as plt

imgR = cv.imread('C:\\Users\\aasid\\Pictures\\unnamed.jpg', 0)

imgL = cv.imread('C:\\Users\\aasid\\Pictures\\unnamed.jpg', 0)
stereo = cv.StereoBM_create(numDisparities = 16,blockSize = 15)

disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, 'gray')
plt.show()
