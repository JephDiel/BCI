# Simple pygame program

# Import and initialize the pygame library
import pygame
from pygame.constants import K_DOLLAR, K_DOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP
from pygame.time import Clock
import random
from ThinkGear import ThinkGear
from collections import deque
import numpy as np

my_device = ThinkGear("COM5")

reads_per_data = 256
raw_data = deque(maxlen=reads_per_data) #192

data_size = 180
time_per_data = 1

lastdata = 0

training_data_input = []
training_data_output = []

data_index = 0

def saveData(dir):
    global data_index
    global training_data_input
    global training_data_output
    training_data_input += [list(raw_data)]
    training_data_output += [[0,0,0,0]]
    training_data_output[-1][dir] = 1
    data_index += 1

pygame.init()



# Set up the drawing window
screen = pygame.display.set_mode([500, 500])
font = pygame.font.Font(pygame.font.get_default_font(), 36)
        

# Run until the user asks to quit
running = True
pos = (250,250)
dir = (0,0)
clock = Clock()
speed = 2

score = 0

foodpos = (250, 100)
while running:
    delta  = 1 / float(clock.tick(reads_per_data))
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and len(raw_data) == reads_per_data:
            if event.key == K_UP:
                print("Up")
                dir = (0, -speed)
                saveData(0)
            if event.key == K_DOWN:
                print("Down")
                dir = (0, speed)
                saveData(1)
            if event.key == K_LEFT:
                print("Left")
                dir = (-speed, 0)
                saveData(2)
            if event.key == K_RIGHT:
                print("Right")
                dir = (speed, 0)
                saveData(3)
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

    # Flip the display
    pygame.display.flip()

    my_device.fetch_data()
    data = my_device.data
    if 'eeg_raw' in data:
        raw_data.append(data['eeg_raw'])
        lastdata = data['eeg_raw']
    else:
        raw_data.append(lastdata)    
    

# Done! Time to quit.
pygame.quit()
my_device.close()

training_data_input = np.array(training_data_input)
training_data_output = np.array(training_data_output)

try:
    old_training_input = np.load("trainingdata/input3.npy")
    old_training_output = np.load("trainingdata/output3.npy")

    training_data_input = np.concatenate((training_data_input, old_training_input))
    training_data_output = np.concatenate((training_data_output, old_training_output))
except:
    print("Old Data Not Found, Overriding")
np.save("trainingdata/input3.npy", training_data_input)
np.save("trainingdata/output3.npy", training_data_output)
print("Added " + str(training_data_input.shape[0]) + " Items with " + str(training_data_input.shape[1]) + " elements each")
