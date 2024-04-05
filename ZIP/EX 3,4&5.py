#EXPERIMENT 3 :

import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    pts1 = np.float32([[0, 260], [640, 260],[0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],[0, 640], [400, 640]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (500, 600))
    cv2.imshow('frame', frame) # Initial Capture
    cv2.imshow('frame1', result) # Transformed Capture
    if cv2.waitKey(24) == 27:
        break
cap.release()
cv2.destroyAllWindows()


#EXPERIMENT 4 :

import cv2 as cv

from matplotlib import pyplot as plt

imgR = cv.imread('C:\\Users\\aasid\\Pictures\\unnamed.jpg', 0)

imgL = cv.imread('C:\\Users\\aasid\\Pictures\\unnamed.jpg', 0)
stereo = cv.StereoBM_create(numDisparities = 16,blockSize = 15)

disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, 'gray')
plt.show()


#EXPERIMENT 5 :

import cv2
import numpy as np
from tqdm import tqdm
# Set the path to the images captured by the left and right cameras
pathL = "./data/stereoL/C:\\Users\\aasid\\Pictures\\test1.jpg"
pathR = "./data/stereoR/C:\\Users\\aasid\\Pictures\\test1.jpg"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1,12)):
    imgL = cv2.imread(pathL+"C:\\Users\\aasid\\Pictures\\test1%d.jpg"%i)
    imgR = cv2.imread(pathR+"C:\\Users\\aasid\\Pictures\\test1%d.jpg"%i)
    imgL_gray = cv2.imread(pathL+"C:\\Users\\aasid\\Pictures\\test1%d.jpg"%i,0)
    imgR_gray = cv2.imread(pathR+"C:\\Users\\aasid\\Pictures\\test1%d.jpg"%i,0)

outputL = imgL.copy()
outputR = imgR.copy()

retR, cornersR = (cv2.findChessboardCorners(outputR,(9,6),None)
retL, cornersL = (cv2.findChessboardCorners(outputL,(9,6),None)

if retR and retL:
    obj_pts.append(objp)
    cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
    cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
    cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
    cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
    cv2.imshow('cornersR',outputR)
    cv2.imshow('cornersL',outputL)
    cv2.waitKey(0)

img_ptsL.append(cornersL)
img_ptsR.append(cornersR)

# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))
