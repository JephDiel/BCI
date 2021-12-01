# from itertools import Predicate, accumulate
from re import VERBOSE
import re
import numpy as np
import numpy
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

from tensorflow.python.keras.backend import conv1d
from tensorflow.python.keras.engine.input_layer import InputLayer
# from tensorflow.core.framework import tensor_description_pb2
# from tensorflow.keras import layers
# from tensorflow.python.keras import activations

input = np.load("trainingdata/input3.npy")
input = input - np.apply_along_axis(lambda a: np.convolve(a, np.ones(11)/11, 'same'), axis=1, arr=input) + 60
output = np.load("trainingdata/output3.npy")
print("Testing with:", input.shape[0], "Data Points with ", input.shape[1], "features")


input = input[..., np.newaxis]
# output = output


models = []
num_of_models = 40
for i in range(num_of_models):
    models += [tf.keras.models.load_model('saved_models/boosting/cnn' + str(i))]


max_accuracy = 0

predictions = []
for i in range(num_of_models):
    predictions += [models[i].predict(input)]


# predictions = np.sum(predictions, axis=0)
# predictions /= np.sum(predictions)

# print(predictions.shape)

correct = 0
incorrect = 0
amount = len(predictions[0])# // 256
confidences = []
predictions_dist = []
correct_dist = []
incorrect_dist = []

for j in range(num_of_models):
    confidences += [[0,0,0,0]]
    predictions_dist += [[0,0,0,0]]
    correct_dist += [[0,0,0,0]]
    incorrect_dist += [[0,0,0,0]]
    correct = 0
    incorrect = 0

    for i in range(amount):
    
        
        max_prediction = max(list(range(4)), key = lambda k: predictions[j][i][k])
        confidence = max(predictions[j][i])
        max_test = max(list(range(4)), key = lambda k: output[i][k])
        confidences[j][max_prediction] += confidence
        predictions_dist[j][max_prediction] += 1

        if (max_prediction == max_test):
            correct += 1
            correct_dist[j][max_prediction] += 1
        else:
            incorrect += 1
            incorrect_dist[j][max_prediction] += 1

# confidences /= confidences[0] + confidences[1] + confidences[2] + confidences[3]
# predictions_dist /= predictions_dist[0] + predictions_dist[1] + predictions_dist[2] + predictions_dist[3]

    accuracy = correct / amount
    print("Model", j)
    print("Accuracy:", accuracy)
    # print(confidences / np.sum(confidences))
    print(predictions_dist[j] / np.sum(predictions_dist[j]))
    print("-----------------------")
    # print(correct_dist / np.sum(correct_dist))
    # print(incorrect_dist / np.sum(incorrect_dist))



predictions = np.sum(predictions, axis=0)

# predictions /= np.sum(predictions, axis=2)
print(predictions[0])

confidences = [0,0,0,0]
predictions_dist = [0,0,0,0]
correct_dist = [0,0,0,0]
incorrect_dist = [0,0,0,0]
correct = 0
incorrect = 0
amount = len(predictions)


for i in range(amount):
    
    
    max_prediction = max(list(range(4)), key = lambda k: predictions[i][k])
    confidence = max(predictions[i])
    max_test = max(list(range(4)), key = lambda k: output[i][k])
    confidences[max_prediction] += confidence
    predictions_dist[max_prediction] += 1

    if (max_prediction == max_test):
        correct += 1
        correct_dist[max_prediction] += 1
    else:
        incorrect += 1
        incorrect_dist[max_prediction] += 1

# confidences /= confidences[0] + confidences[1] + confidences[2] + confidences[3]
# predictions_dist /= predictions_dist[0] + predictions_dist[1] + predictions_dist[2] + predictions_dist[3]

accuracy = correct / amount
print("Overall")
print("Accuracy:", accuracy)
# print(confidences / np.sum(confidences))
print(predictions_dist / np.sum(predictions_dist))