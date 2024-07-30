# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:45:58 2023

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def convolve2d(X, W):
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1+m1-1,n2+m2-1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1,j:j+m2] += X[i,j]*W
    return Y

img = mpimg.imread('lenna.png')
plt.imshow(img)
plt.show()

bw = img.mean(axis=2)
plt.imshow(bw, cmap = 'gray')
plt.show()

W = np.zeros((20,20))

for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50)

plt.imshow(W, cmap = 'gray')
plt.show()

out = convolve2d(bw, W, mode = 'same')

out.shape

plt.imshow(out, cmap = 'gray')
plt.show()

out3 = np.zeros(img.shape)

for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i]/255, W, mode = 'same')
    
plt.imshow(out3)
plt.show()
