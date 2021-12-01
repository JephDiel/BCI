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
# input = input - np.apply_along_axis(lambda a: np.convolve(a, np.ones(11)/11, 'same'), axis=1, arr=input) + 60
output = np.load("trainingdata/output3.npy")
print("Testing with:", input.shape[0], "Data Points with ", input.shape[1], "features")


input = input[..., np.newaxis]
# output = output


# model = tf.keras.models.load_model('saved_models/bestModelconvolved3')

# model.summary()

# max_accuracy = 0

# predictions = model.predict(input)

model = [0,0,0,0]
predictions = np.zeros((len(input), 4))

for i in range(4):
    model[i] = tf.keras.models.load_model('saved_models/perclass/CNN' + str(i))
    print(model[i].predict(input))

# predictions = np.transpose(predictions).squeeze()

print(predictions)




# predictions = predictions * [0.08718487,  0.50945378, 0.18592437, 0.21743697]
weights = [0.08718487,  0.50945378, 0.18592437, 0.21743697]

# print(predictions_weighted[10][0] / predictions[10][0])

correct = 0
incorrect = 0
amount = len(predictions)# // 256
confidences = [0,0,0,0]
predictions_dist = [0,0,0,0]
correct_dist = [0,0,0,0]
incorrect_dist = [0,0,0,0]

for i in range(amount):
    max_prediction = max(list(range(4)), key = lambda j: predictions[i][j])
    confidence = max(predictions[i])
    max_test = max(list(range(4)), key = lambda j: output[i][j])
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
print("Accuracy:", accuracy)
print(confidences / np.sum(confidences))
print(predictions_dist / np.sum(predictions_dist))
print(correct_dist)
print(incorrect_dist)
print(np.divide(correct_dist, incorrect_dist))


