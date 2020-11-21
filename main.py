import serial
import time

from ThinkGear import ThinkGear
import pygame
from pygame.locals import *
import random
from collections import deque
import numpy as np

my_device = ThinkGear("COM5")
# try:
#     while True:
#         my_device.fetch_data()
#         data = my_device.data
#         if 'eeg_raw' in data:
#             print(data['eeg_raw'])
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     my_device.close()
#     time.sleep(0.5)
#     exit()

pygame.init()

gameDisplay=pygame.display.set_mode((600,600))
pygame.display.set_caption("Platypus")

white=255,255,255
black=0,0,0
red=255,0,0

images = [[], [], [], []]




for i in range(0, 1):
    images[0] += [pygame.image.load("data/images/up/up" + str(i) + ".png")]
    images[1] += [pygame.image.load("data/images/down/down" + str(i) + ".png")]
    images[2] += [pygame.image.load("data/images/left/left" + str(i) + ".png")]
    images[3] += [pygame.image.load("data/images/right/right" + str(i) + ".png")]

reads_per_data = 256

raw_data = deque(maxlen=int(reads_per_data * 1)) #192
counter = 0
lastdata = 0

data_size = 240
time_per_data = 1

training_data_input = np.zeros((data_size, reads_per_data))
training_data_output = np.zeros((data_size, 4))


for sample in range(data_size):
    dir = random.randrange(4)
    gameDisplay.blit(images[dir][random.randrange(len(images[dir]))], (0,0))
    pygame.display.update()
    time_save = time.time()
    for reads in range(reads_per_data):
        delay = time.time()
        my_device.fetch_data()
        data = my_device.data
        if 'eeg_raw' in data:
            raw_data.append(data['eeg_raw'])
            lastdata = data['eeg_raw']
        else:
            raw_data.append(lastdata)
        

        time.sleep(time_per_data/reads_per_data - (time.time() - delay))
        delay = time.time()

    # print(list(raw_data))
    training_data_input[sample] = list(raw_data)
    training_data_output[sample][dir] = 1
    print(time.time() - time_save)

print(training_data_input, training_data_input.shape)
# np.save("trainingdata/input.npy", training_data_input)
# np.save("trainingdata/output.npy", training_data_output)
pygame.quit()
quit()