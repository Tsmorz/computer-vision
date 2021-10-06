import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
from numpy import linalg
from scipy.signal import argrelextrema

def GaussianBlur(sigma):
    spread = sigma*7  # 99.73% of values
    x = range(spread)
    x = x - np.median(x)
    fxn = 1/sigma*np.sqrt(2*np.pi) * np.exp(-0.5*(x/sigma)**2)
    horz = fxn.reshape(spread,1)
    vert = fxn.reshape(1,spread)
    g = np.dot(horz,vert)
    return g

def ImageSlopes(img):
    r,c = img.shape
    Iy = np.diff(img,axis=0)
    pad = np.zeros((1,c))
    Iy = np.vstack((Iy, pad))
    Ix = np.diff(img,axis=1)
    pad = np.zeros((r,1))
    Ix = np.hstack((Ix, pad))
    return Ix, Iy

def DoubleSlopes(Ix, Iy):
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2
    return Ixx, Ixy, Iyy



# import image
file = 'tomato.jpg'
img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

'''
# test image
gray = np.zeros((200,300))
gray[50:-50,100:-100] = 255
'''

### Harris Corner Detection

# blur image
sigma = 3
blur = cv.GaussianBlur(gray, (0, 0), sigma)

# image derivatives
gIx, gIy = ImageSlopes(blur)

# M values
gIxx, gIxy, gIyy = DoubleSlopes(gIx, gIy)

# compute R's
k = 0.05
R = gIxx*gIyy - k*(gIxx+gIyy)**2

# threshold
Rnorm = R.copy()
Rnorm[Rnorm <= 0.05*np.max(Rnorm)] = 0

# local maxima
max_ind = argrelextrema(Rnorm, np.greater)
print(max_ind)

print(gray.shape)
print(Rnorm.shape)
print(gray)
print(blur)
mask = Rnorm + gray

# Figures
fig = plt.figure()

plt.subplot(4,2,1)
plt.imshow(gray,'gray')

plt.subplot(4,2,2)
plt.imshow(blur)

plt.subplot(4,2,3)
plt.imshow(gIx)

plt.subplot(4,2,4)
sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
plt.imshow(sobelx)

plt.subplot(4,2,5)
plt.imshow(gIy)

plt.subplot(4,2,6)
plt.imshow(gIyy)

plt.subplot(4,2,7)
plt.imshow(R)

plt.subplot(4,2,8)
plt.imshow(Rnorm)

plt.show()
