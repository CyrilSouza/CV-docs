#EXPERIMENT 7 :

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread 
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndimage 

imageFile = "C:/Users/Rajesh/Desktop/3.jpg" 
mat = imread(imageFile) 
mat = mat[:,:,0] 
rows, cols = mat.shape 
xv, yv = np.meshgrid(range(cols), range(rows)[::-1]) 

blurred = ndimage.gaussian_filter(mat, sigma=(5, 5), order=0) 
fig = plt.figure(figsize=(6,6)) 

ax = fig.add_subplot(221) 
ax.imshow(mat, cmap='gray') 

ax = fig.add_subplot(222, projection='3d') 
ax.elev= 75 
ax.plot_surface(xv, yv, mat) 

ax = fig.add_subplot(223) 
ax.imshow(blurred, cmap='gray') 

ax = fig.add_subplot(224, projection='3d') 
ax.elev= 75 
ax.plot_surface(xv, yv, blurred) 
plt.show()






