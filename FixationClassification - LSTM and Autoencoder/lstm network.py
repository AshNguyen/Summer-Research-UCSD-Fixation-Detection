import numpy as np
import math
from keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Adadelta
import random

timestep = 64
inputs = Input(shape=(timestep, 6))
x = LSTM(100, return_sequences=True)(inputs)
x = Dropout(0.3)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)
pred = Dense(8, activation='softmax')(x)

model = Model(inputs=inputs,
              outputs=pred)

optimizer = RMSprop(lr=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])

train_input = np.load('/Users/ash/Downloads/DATA_RNN/rnn_train_gaze_pupil')
train_label = np.load('/Users/ash/Downloads/DATA_RNN/rnn_train_label')

X_train = []
y_train = []
for i in range(timestep,train_input.shape[0]):
    X_train.append(train_input[i-timestep:i, :])
    y_train.append(train_label[i,:])

X_train = np.array(X_train)
y_train = np.array(y_train)

shuffling = np.random.permutation(60000)
X_train, y_train = X_train[shuffling], y_train[shuffling]

model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2)