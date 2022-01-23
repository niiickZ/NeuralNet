""" getting started to AutoEncoder
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize to [-1.0, 1.0]
X_train = X_train - np.mean(X_train)
X_train = X_train / np.max(np.abs(X_train))
X_test = X_test - np.mean(X_test)
X_test = X_test / np.max(np.abs(X_test))

input_img = Input(shape=(784,))

# encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(2)(encoded)

# decoder
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoder_output = Dense(784, activation='tanh')(decoded)

AutoEncoder = Model(inputs=input_img, outputs=decoder_output)
Encoder = Model(inputs=input_img, outputs=encoder_output)

AutoEncoder.compile(optimizer='adam', loss='mse')

AutoEncoder.fit(X_train, X_train, epochs=20, batch_size=128)

encoded_imgs = Encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=Y_test)
plt.show()