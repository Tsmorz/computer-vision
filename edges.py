import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

file = 'Small Bend.jpg'
img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)


fig = plt.figure()

x = np.diff(img,axis=0)
y = np.diff(img,axis=1)

xx = np.diff(x,axis=0)
yy = np.diff(y,axis=1)
xy = np.diff(x,axis=1)

cv2.imshow('Grayscale', img)
cv2.waitKey(0) 

def gaussian(image):
    mu, sigma = 0, 0.1
    return image