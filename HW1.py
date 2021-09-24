import numpy as np

RGB = np.array([200, 50, 100]).T
max = 255
RGBp = RGB/max
R = RGBp[0]
G = RGBp[1]
B = RGBp[2]

CMY = 1-RGB/max
print('CMY')
print(CMY)
print()

YIQmat = [[0.299, 0.587, 0.114],
          [0.596, -0.275, -0.321],
          [0.212, -0.532, 0.311]]
YIQ = np.dot(YIQmat,RGBp)
print('YIQ')
print(YIQ)
print()

theta = np.arccos( 0.5*((R-G)+(R-B)) / ((R-G)**2+(R-B)*(G-B) )**0.5 )
if B<= G:
    H = theta
else:
    H = 2*np.pi-theta
S = 1-3/(R+G+B)*np.min([R,G,B])
I = 1/3*(R+G+B)
HSI = np.array([H, S, I]).T
print('HSI')
print(HSI)
print()