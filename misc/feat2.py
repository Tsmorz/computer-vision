import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
from numpy import linalg
from scipy.signal import argrelextrema


# import image
file = 'Small Bend.jpg'
img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

### Harris Corner Detection

# blur image
sigma = 3
blur = cv.GaussianBlur(gray, (0, 0), sigma)

ksize = 5
Ix = cv.Sobel(gray,cv.CV_64F,1,0,ksize)
Iy = cv.Sobel(gray,cv.CV_64F,0,1,ksize)

Ixx = Ix**2
Ixy = Ix*Iy
Iyy = Iy**2

# compute R's
k = 0.04
R = Ixx*Iyy - k*(Ixx+Iyy)**2

# threshold
Rnorm = R.copy()
Rnorm = Rnorm - Rnorm.min()
Rnorm = 255*Rnorm/Rnorm.max()
Rnorm[Rnorm>=0.16*Rnorm.max()] = 255
print(Rnorm.max())
print(Rnorm.min())
# local maxima
max_ind = argrelextrema(Rnorm, np.greater)
print(max_ind)

mask = Rnorm + gray

# Figures
fig = plt.figure()

plt.subplot(4,2,1)
plt.imshow(gray,'gray')
plt.subplot(4,2,2)
plt.imshow(blur)

plt.subplot(4,2,3)
plt.imshow(Ix)
plt.subplot(4,2,4)
plt.imshow(Ixx)

plt.subplot(4,2,5)
plt.imshow(Iy)
plt.subplot(4,2,6)
plt.imshow(Iyy)

plt.subplot(4,2,7)
plt.imshow(R)
plt.subplot(4,2,8)
plt.imshow(Rnorm)

plt.show()