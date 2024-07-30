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


W = np.zeros((20,20))

for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50)

plt.imshow(W, cmap = 'gray')

out = convolve2d(bw, W, mode = 'same')

out.shape

plt.imshow(out, cmap = 'gray')

out3 = np.zeros(img.shape)

for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i]/255, W, mode = 'same')
    
plt.imshow(out3)
