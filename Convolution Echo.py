# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:45:58 2023

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.io.wavfile import write

s_rate = 44100
spf = wave.open('see-this-this-is-my-boomstick!.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'int16')
print('numpy signal shape', signal.shape )

plt.plot(signal)
plt.title('sound without echo')
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print('noecho signal:', noecho.shape)

noecho = noecho.astype(np.int16)
write('noecho.wav', s_rate, noecho)

plt.plot(signal)
plt.title('sound without echo')
plt.show()

filt = np.zeros(s_rate)
filt[0] = 1
filt[int(s_rate*0.3)] = 0.6
filt[int(s_rate*0.5)] = 0.3
filt[int(s_rate*0.8)] = 0.2
filt[s_rate - 1] = 0.1

out = np.convolve(signal, filt)
out = out.astype(np.int16)

plt.plot(out)
plt.title('sound with echo')
plt.show()

write('out.wav', s_rate, out)
