# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:45:58 2023

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.image as mpimg

img = mpimg.imread('lenna.png')
plt.imshow(img)
bw = img.mean(axis=2)
plt.imshow(bw, cmap = 'gray')

Hx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
    ], dtype=np.float32)

Hy = Hx.T

Gx = convolve2d(bw, Hx)

plt.imshow(Gx, cmap = 'gray')

Gy = convolve2d(bw, Hy)

plt.imshow(Gy, cmap = 'gray')

G = np.sqrt(Gx*Gx + Gy*Gy)

plt.imshow(G, cmap = 'gray')

theta = np.arctan2(Gy, Gx)

plt.imshow(theta, cmap = 'gray')
