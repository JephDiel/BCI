# from itertools import Predicate, accumulate
from re import VERBOSE
import re
import numpy as np
import numpy
from tensorflow import keras
import os

from tensorflow.python.keras.backend import conv1d
from tensorflow.python.keras.engine.input_layer import InputLayer
# from tensorflow.core.framework import tensor_description_pb2
# from tensorflow.keras import layers
# from tensorflow.python.keras import activations

input = np.load("trainingdata/input3.npy")
input = np.apply_along_axis(lambda a: np.convolve(a, np.ones(20), 'valid'), axis=1, arr=input)
output = np.load("trainingdata/output3.npy")
print("Learning with:", input.shape[0], "Data Points with ", output.shape[0], "outputs")

split_mark = int((len(input)) * 0.99)
print("Split Mark:", split_mark)

train_input = input[:split_mark]
train_output = output[:split_mark]


test_input = input[split_mark:]
test_output = output[split_mark:]

# test_input = np.load("trainingdata/input.npy")
# test_output = np.load("trainingdata/output.npy")

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
        # keras.layers.Dense(units=256, input_dim=256, activation='relu'),
        keras.layers.InputLayer(237),
        # keras.layers.Conv1D(256, 2048, activation='relu'),
        keras.layers.Dense(units=2048, activation='relu'),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Dense(units=4, activation='softmax')
        ])

# model.build(input_shape=(128, 1))
# model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_dataset)
results = []


# opt = keras.optimizers.Adam(learning_rate=0.01)      
model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

max_accuracy = 0

for epochs in range(0, 200, 1):

    model.fit(train_input, train_output, epochs=1, verbose=0)

    predictions = model.predict(test_input)

    correct = 0
    incorrect = 0
    amount = len(predictions)# // 256

    for i in range(amount):

        # prediction = [0,0,0,0]

        # truth = max_test = max(list(range(4)), key = lambda j: test_output[i * 256][j])

        # for window in range(256):
        #     max_prediction = max(list(range(4)), key = lambda j: predictions[(i * 256) + window][j])
        #     confidence = max(predictions[(i * 256) + window])
        #     max_test = max(list(range(4)), key = lambda j: test_output[(i * 256) + window][j])
        #     if (max_test != truth):
        #         print("Error Parsing Window, inconsistent truth")
        #         exit()
        #     for k in range(4):
        #         prediction[k] += predictions[(i * 256) + window][k] if predictions[(i * 256) + window][k] > 0.2 else 0
            # print("Predicted:", max_prediction)
            # print("Actual: ", max_test)
            # print("Confidence:", confidence)

        max_prediction = max(list(range(4)), key = lambda j: predictions[i][j])
        confidence = max(predictions[i])
        max_test = max(list(range(4)), key = lambda j: test_output[i][j])

        # normalizing_factor = sum(prediction)
        # for k in range(4):
        #     prediction[k] /= normalizing_factor
        
        # max_prediction = max(list(range(4)), key = lambda j: prediction[j])
        # if max_prediction == max_test:
        #     correct += 1
        # else:
        #     incorrect += 1

        if (max_prediction == max_test):
            correct += 1
        else:
            incorrect += 1
        # print("Truth", truth)
        # print("Prediction:", max_prediction)
        # print("Confidence:", prediction[max_prediction])
        # print("Distribution:", prediction)
        # print("--------------------------")
    accuracy = correct / amount
    print("Accuracy:", accuracy)
    print("Epochs:", epochs)
    print("---------------------")
    
    print("---------------------")
    print("---------------------")
    print("Max Accuracy:", max_accuracy)
    print("Epochs:", epochs)
    print("---------------------")
    print("---------------------")
    results.append((max_accuracy, epochs))

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model.save("saved_models/bestModelconvolved")
        readme = open("saved_models/bestModel3/README.txt", "w+")
        readme.write("Current Best BCI Prediction Model:\n" + "\t accuracy: " + str(max_accuracy) + "\n\tepochs: " + str(epochs))
        readme.close()
for result in results:
    print("Max Accuracy:", result[0], "in", result[1], "epochs")
