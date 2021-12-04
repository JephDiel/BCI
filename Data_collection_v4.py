# Simple pygame program

# Import and initialize the pygame library
from numpy.lib.type_check import imag
import pygame
from pygame.constants import K_DOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP
from pygame.time import Clock
from ThinkGear import ThinkGear
import random
import numpy as np
from variables import *

my_device = ThinkGear("COM5")

reads_per_second = 256

lastdata = 0

training_data_input = np.array([], dtype=np.int32)
training_data_output = np.array([], dtype=np.int32)

data_index = 0

pygame.init()



# Set up the drawing window
screen = pygame.display.set_mode([500, 500])
font = pygame.font.Font(pygame.font.get_default_font(), 36)

images = [0,0,0,0]


images[UP] = pygame.image.load("./data/images/up.png")
images[DOWN] = pygame.image.load("./data/images/down.png")
images[LEFT] = pygame.image.load("./data/images/left.png")
images[RIGHT] = pygame.image.load("./data/images/right.png")

for i in range(len(images)):
    images[i] = pygame.transform.scale(images[i], (25, 25))
# Run until the user asks to quit
running = True
pos = (250,250)
dir = NOTHING
pos_delta = (0,0)
clock = Clock()
speed = 2

score = 0

foodpos = (250, 100)

direction_texts = ["","", "", "", ""]
direction_texts[UP] = "Up"
direction_texts[DOWN] = "Down"
direction_texts[LEFT] = "Left"
direction_texts[RIGHT] = "Right"



while running:
    delta_time  = 1 / float(clock.tick(reads_per_second))
    # Did the user click the window close button?


    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        # print("up")
        dir = UP
    elif keys[K_DOWN]:
        # print("down")
        dir = DOWN
    elif keys[K_LEFT]:
        # print("left")
        dir = LEFT
    elif keys[K_RIGHT]:
        # print("right")
        dir = RIGHT
    else:
        # print("nothing")
        dir = NOTHING

    if dir == UP:
        pos_delta = (0, -speed)
    elif dir == DOWN:
        pos_delta = (0, speed)
    elif dir == LEFT:
        pos_delta = (-speed, 0)
    elif dir == RIGHT:
        pos_delta = (speed, 0)
    elif dir == NOTHING:
        pos_delta = (0,0)
    


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    # Fill the background with white
    screen.fill((50, 50, 50))
    
    if len(training_data_input) > reads_per_second:
        pygame.draw.circle(screen, (0, 150, 0), foodpos, 5)

        if dir == NOTHING:
            pygame.draw.circle(screen, (255, 217, 92), pos, 10)
        else:
            screen.blit(images[dir], (pos[0]-12.5, pos[1]-12.5))
        x = (pos[0] + (pos_delta[0] * delta_time)) % 500
        y = (pos[1] + (pos_delta[1] * delta_time)) % 500
        pos = (x,y)

        text_surface = font.render(direction_texts[dir], True, (250, 250, 250))
        text_rect = text_surface.get_rect(center = (250, 20))
    
        screen.blit(text_surface, text_rect)

    if ((pos[0]-foodpos[0])**2 + (pos[1] - foodpos[1]) ** 2) < 225:
        score += 1
        
        while ((pos[0]-foodpos[0])**2 + (pos[1] - foodpos[1]) ** 2) < 225:
            foodpos = (random.randint(10, 490), random.randint(40, 490))

    # Flip the display
    pygame.display.flip()

    my_device.fetch_data()
    data = my_device.data
    if 'eeg_raw' in data:
        lastdata = data['eeg_raw']

    training_data_input = np.append(training_data_input, lastdata)
    training_data_output = np.append(training_data_output, dir)
    

# Done! Time to quit.
pygame.quit()
my_device.close()

training_data_input = np.array(training_data_input, dtype=np.uint8)
training_data_output = np.array(training_data_output, dtype=np.uint8)

print("Adding " + str(training_data_input.shape[0]) + " Items")


# try:
#     old_training_data_input = np.load("trainingdata/input3.npy")
#     old_training_data_output = np.load("trainingdata/output3.npy")

#     training_data_input = np.concatenate((old_training_data_input, training_data_input))
#     training_data_output = np.concatenate((old_training_data_output, training_data_output))

# except:
#     print("Old Data Not Found, Overriding")

# np.save("trainingdata/input_v4.npy", training_data_input)
# np.save("trainingdata/output_v4.npy", training_data_output)

# print("Total " + str(training_data_input.shape[0]) + " Items")
