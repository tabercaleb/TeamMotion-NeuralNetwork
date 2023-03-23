import numpy as np
from numpy import loadtxt
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from os import path

"""
It is possible to loop the training
Recommended to only to it at a maximum of 3 times
3 is usually when a peak is hit
Uncomment the while statement and indent everything down to run increment
"""
run = 0
runMax = 3
testing_accuracy = []
#while run < runMax:


dataset = loadtxt('Squat_Filtered.csv', delimiter=',')
dataset_pd = pd.DataFrame(dataset)

training_data = dataset_pd.sample(frac=0.7)
testing_data = dataset_pd.drop(training_data.index)

print(f"Number of training examples: {training_data.shape[0]}")
print(f"Number of testing examples: {testing_data.shape[0]}")

training_data = training_data.to_numpy()
testing_data = testing_data.to_numpy()

X = training_data[:, 0:34]
Y = training_data[:, 34]

test_X = testing_data[:, 0:34]
test_Y = testing_data[:, 34]

model = Sequential()

if path.exists("model.h5"):
    print("Using pre existing model")
    model = tf.keras.models.load_model('model.h5')
    model.summary()
else:
    print("Creating new model")
    model = Sequential()

    model.add(Dense(17, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.predict(X)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

history = model.fit(X, Y, epochs=500, batch_size=20, validation_split=.2, shuffle=True)

_, accuracy = model.evaluate(X, Y)

predictions = model.predict(test_X)

fl = len(test_Y)
true_positive = 0
false_positive = 0

for i in range(fl):
    predicted_label = np.argmax(predictions[i])
    if test_Y[i] == predicted_label:
        true_positive += 1
    else:
        false_positive += 1

t_accuracy = true_positive / len(test_Y)

model.save('model.h5')

print("Model Saved")

print(f"Testing accuracy: {t_accuracy}")
#testing_accuracy.append(t_accuracy) # uncomment for looping

    #run += 1 # uncomment for looping

"""
can be used with both single runs and looped runs
only prints out the last run per loop
keep outside of while loop
"""
#print(testing_accuracy) # uncomment for looping
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title("Accuracy and Loss")
plt.ylim([0, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy and Loss')
plt.legend()
plt.grid(True)
plt.show()