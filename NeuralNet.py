# from itertools import Predicate, accumulate
from random import shuffle
from re import VERBOSE
import re
import numpy as np
import numpy
from numpy.core.defchararray import mod
from numpy.core.fromnumeric import amax, size
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.module.module import valid_identifier
# from tensorflow.core.framework import tensor_description_pb2
# from tensorflow.keras import layers
# from tensorflow.python.keras import activations

input = np.load("trainingdata/input2.npy")
print("Learning with:", input.shape[0], "Data Points")

output = np.load("trainingdata/output2.npy")

split_mark = int((len(input) // 256) * 0.7) * 256

train_input = input[:split_mark]
train_output = output[:split_mark]

test_input = input[split_mark:]
test_output = output[split_mark:]
# print(test_input[np.newaxis,...].shape)
# print(train_input.shape, train_output.shape)
# train = np.vstack([test_input[np.newaxis,...],test_output[np.newaxis,...]])#np.array([train_input.T, train_output.T]).T#np.dstack([train_input, [train_output]])
# test = np.dstack([test_input, [test_output]])

# print(train.shape)
# print(train)
# print(output)
# exit()

# train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_output))

model = keras.Sequential([
    keras.layers.Dense(units=256, input_dim=256, activation='relu'),
    keras.layers.Dense(units=192, activation='relu'),
    keras.layers.Dense(units=144, activation='relu'),
    keras.layers.Dense(units=108, activation='relu'),
    keras.layers.Dense(units=81, activation='relu'),
    keras.layers.Dense(units=60, activation='relu'),
    keras.layers.Dense(units=45, activation='relu'),
    keras.layers.Dense(units=34, activation='relu'),
    keras.layers.Dense(units=25, activation='relu'),
    keras.layers.Dense(units=25, activation='relu'),
    keras.layers.Dense(units=19, activation='relu'),
    keras.layers.Dense(units=12, activation='relu'),
    keras.layers.Dense(units=4, activation='softmax')
])

# model.build(input_shape=(128, 1))
# model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_dataset)
results = []
for epochs in range(0, 100, 1):
    max_accuracy = 0
    
    for i in range(0, 100):
        model = keras.Sequential([
        keras.layers.Dense(units=256, input_dim=256, activation='relu'),
        keras.layers.Dense(units=192, activation='relu'),
        keras.layers.Dense(units=144, activation='relu'),
        keras.layers.Dense(units=108, activation='relu'),
        keras.layers.Dense(units=81, activation='relu'),
        keras.layers.Dense(units=60, activation='relu'),
        keras.layers.Dense(units=45, activation='relu'),
        keras.layers.Dense(units=34, activation='relu'),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=19, activation='relu'),
        keras.layers.Dense(units=12, activation='relu'),
        keras.layers.Dense(units=4, activation='softmax')
        ])
        # opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
            
        model.fit(train_input, train_output, epochs=epochs, verbose=0)

        # predictions = model.evaluate(test_dataset)
        predictions = model.predict(test_input)
        # print(predictions)
        # print(predictions[1,1])
        # print(test_output)
        # exit()


        correct = 0
        incorrect = 0
        amount = len(predictions) // 256

        for i in range(amount):

            prediction = [0,0,0,0]

            truth = max_test = max(list(range(4)), key = lambda j: test_output[i * 256][j])

            for window in range(256):
                max_prediction = max(list(range(4)), key = lambda j: predictions[(i * 256) + window][j])
                confidence = max(predictions[(i * 256) + window])
                max_test = max(list(range(4)), key = lambda j: test_output[(i * 256) + window][j])
                if (max_test != truth):
                    print("Error Parsing Window, inconsistent truth")
                    exit()
                for k in range(4):
                    prediction[k] += predictions[(i * 256) + window][k]
                # print("Predicted:", max_prediction)
                # print("Actual: ", max_test)
                # print("Confidence:", confidence)
            
            normalizing_factor = sum(prediction)
            for k in range(4):
                prediction[k] /= normalizing_factor
            
            max_prediction = max(list(range(4)), key = lambda j: prediction[j])
            if max_prediction == max_test:
                correct += 1
            else:
                incorrect += 1
            # print("Truth", truth)
            # print("Prediction:", max_prediction)
            # print("Confidence:", prediction[max_prediction])
            # print("Distribution:", prediction)
            # print("--------------------------")
        accuracy = correct / amount
        # print("Accuracy:", accuracy)
        # print("Epochs:", epochs)
        # print("---------------------")
        max_accuracy = max(max_accuracy, accuracy)
    print("Accuracy:", max_accuracy)
    print("Epochs:", epochs)
    print("---------------------")
    results.append((max_accuracy, epochs))

for result in results:
    print("Max Accuracy:", result[0], "in", result[1], "epochs")
