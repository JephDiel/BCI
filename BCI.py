import tensorflow as tf
from tensorflow import keras
import time

from ThinkGear import ThinkGear
import pygame
from pygame.time import Clock
from pygame.locals import *
import random
from collections import deque
import numpy as np

my_device = ThinkGear("COM5")

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('saved_models/bestModelconvolved')

# Show the model architecture
model.summary()

pygame.init()

gameDisplay=pygame.display.set_mode((600,600))
pygame.display.set_caption("Platypus")

white=255,255,255
black=0,0,0
red=255,0,0

images = [[], [], [], []]




i = 1
images[0] += [pygame.image.load("data/images/up/up" + str(i) + ".png")]
images[1] += [pygame.image.load("data/images/down/down" + str(i) + ".png")]
images[2] += [pygame.image.load("data/images/left/left" + str(i) + ".png")]
images[3] += [pygame.image.load("data/images/right/right" + str(i) + ".png")]

reads_per_data = 256

raw_data = deque(maxlen=int(reads_per_data * 1)) #192
lastdata = 0

data_size = 180
time_per_data = 1

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])
font = pygame.font.Font(pygame.font.get_default_font(), 36)
        

# Run until the user asks to quit
running = True
pos = (250,250)
dir = (0,0)
clock = Clock()
speed = 10

score = 0
foodpos = (250, 100)

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

running = True
evaluation = [0,0,0,0]

while running:
    evaluation = model.predict(np.array([list(raw_data)])[..., np.newaxis])[0]
    # print(new_evaluation)
    # for i in range(4):
    #     evaluation[i] = (0.99 * evaluation[i]) + (0.01 * new_evaluation[i])
    direction = max(list(range(4)), key = lambda j: evaluation[j])
    delta  = 1 / float(clock.tick(reads_per_data))
    # Did the user click the window close button?
    if direction == 0:
        print("Up")
        dir = (0, -speed)
    if direction == 1:
        print("Down")
        dir = (0, speed)
    if direction == 2:
        print("Left")
        dir = (-speed, 0)
    if direction == 3:
        print("Right")
        dir = (speed, 0)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    # Fill the background with white
    screen.fill((50, 50, 50))
    if(len(raw_data) == reads_per_data):
        pygame.draw.circle(screen, (0, 150, 0), foodpos, 5)

        # Draw a solid blue circle in the center
        pygame.draw.circle(screen, (200, 200, 0), pos, 10)
    x = (pos[0] + (dir[0] * delta)) % 500
    y = (pos[1] + (dir[1] * delta)) % 500
    pos = (x,y)

    text_surface = font.render("Score: " + str(score), True, (250, 250, 250))
    screen.blit(text_surface, dest=(10,10))

    if ((pos[0]-foodpos[0])**2 + (pos[1] - foodpos[1]) ** 2) < 225:
        score += 1
        print("Score")
        
        while ((pos[0]-foodpos[0])**2 + (pos[1] - foodpos[1]) ** 2) < 225:
            foodpos = (random.randint(10, 490), random.randint(40, 490))
    
    # gameDisplay.blit(images[dir][random.randrange(len(images[dir]))], (0,0))
    # pygame.display.update()
    time_save = time.time()
    pygame.display.flip()

    my_device.fetch_data()
    data = my_device.data
    if 'eeg_raw' in data:
        raw_data.append(data['eeg_raw'])
        lastdata = data['eeg_raw']
    else:
        raw_data.append(lastdata)
    
    for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

    time.sleep(time_per_data/reads_per_data - (time.time() - time_save))

pygame.quit()
quit()