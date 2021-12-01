# from itertools import Predicate, accumulate
from re import VERBOSE
import re
import numpy as np
import numpy
from numpy.core.numeric import outer
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import os

from tensorflow.python.keras.backend import conv1d
from tensorflow.python.keras.engine.input_layer import InputLayer
# from tensorflow.core.framework import tensor_description_pb2
# from tensorflow.keras import layers
# from tensorflow.python.keras import activations

input = np.load("trainingdata/input3.npy")
# input = input[..., 128:]
# input = input - np.apply_along_axis(lambda a: np.convolve(a, np.ones(11)/11, 'same'), axis=1, arr=input) + 60
output = np.load("trainingdata/output3.npy")
output = output[:, 3]
print("Learning with:", input.shape[0], "Data Points with ", input.shape[1], "features")

split_mark = int((len(input)) * 0.7)
print("Split Mark:", split_mark)

train_input = input[:split_mark][..., np.newaxis]
train_output = output[:split_mark]


test_input = input[split_mark:][..., np.newaxis]
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

num_filters = 32
size_kernal = 3

for boost in range(1):

    model = keras.Sequential([
            # keras.layers.InputLayer(237),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu', input_shape=(256, 1)),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.Conv1D(filters=num_filters, kernel_size=size_kernal, padding="same", activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            # keras.layers.Dense(units=8*64, activation='relu'),
            # keras.layers.Dense(units=32, activation='relu'),
            keras.layers.Dense(units=16, activation='relu', use_bias = True),
            keras.layers.Dense(units=4, activation='relu'),
            keras.layers.Dense(units=1, activation='relu')
            ])


    # opt = keras.optimizers.Adam(learning_rate=0.0001)      
    model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    max_accuracy = 0


    checkpoint = ModelCheckpoint("saved_models/perclass/CNN3", monitor='val_accuracy', verbose=1, \
                                save_best_only=True, save_weights_only=False, \
                                mode='auto', save_frequency=1)


    model.fit(train_input, train_output, epochs=20, verbose=1, shuffle=True, batch_size=8, validation_data=(test_input, test_output), callbacks=[checkpoint])

    exit()

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
    # print("Epochs:", epochs)
    # print("---------------------")

    # print("---------------------")
    # print("---------------------")
    # print("Max Accuracy:", max_accuracy)
    # # print("Epochs:", epochs)
    # print("---------------------")
    # print("---------------------")
    # results.append((max_accuracy, epochs))

    # if accuracy > max_accuracy:
    #     max_accuracy = accuracy
    #     model.save("saved_models/bestModelconvolved")
    #     readme = open("saved_models/bestModel3/README.txt", "w+")
    #     readme.write("Current Best BCI Prediction Model:\n" + "\t accuracy: " + str(max_accuracy) + "\n\tepochs: " + str(epochs))
    #     readme.close()

    # for result in results:
        # print("Max Accuracy:", result[0], "in", result[1], "epochs")
