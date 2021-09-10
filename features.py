import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

file = 'Small Bend.jpg'
file = 'checkers.png'
file = 'images.jpeg'
img = cv.imread(file)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(img[:,0], img[:,1], img[:,2])
plt.show()

print(img.shape)
x = np.diff(img,axis=0)
y = np.diff(img,axis=1)

xx = np.diff(x,axis=0)
yy = np.diff(y,axis=1)
xy = np.diff(x,axis=1)

plt.imshow(y)

plt.show()