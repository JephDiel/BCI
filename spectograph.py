import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

input = np.load("trainingdata/input.npy")

f, t, Sxx = signal.spectrogram(input[50])

plt.imshow(Sxx)
plt.show()




