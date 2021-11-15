import numpy as np
import sys
from pylab import *
import wave

input = np.load("trainingdata/input3.npy")
output = np.load("trainingdata/output3.npy")

input_smoothed = input - np.apply_along_axis(lambda a: np.convolve(a, np.ones(11)/11, 'same'), axis=1, arr=input) + 60

print(len(input[0]), len(input_smoothed[0]))

print(np.sum(output, axis=0) / np.sum(output))

print(input)
print("----------------")
print(input[..., np.newaxis])


output = np.load("trainingdata/output3.npy")
print("Analyzing:", input.shape[1], "Data Points with ", output.shape[0], "outputs")


# count = 0
# j = 0
# waves = []
# while count < 10 and j < len(input):
#     if (output[j][0] == 1):
#         waves += [input_smoothed[j]]
#         count += 1
#     j += 1

waves = [[],[]]

waves[0] = input[100]

# sensitivity = 10

# waves[1] = np.convolve(waves[0], np.ones(sensitivity), 'valid') / sensitivity

waves[1] = input_smoothed[100]

f = 256 #framerate

i = 1
for wave in waves:
    subplot(len(waves), 1, i)
    plot(wave)
    i += 1

# subplot(212)
# spectrogram = specgram(wave1, Fs = f, scale_by_freq=True,sides='default')

show()
