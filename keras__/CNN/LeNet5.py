"""
    CNN getting started example —— LeNet-5
    p.s. The original paper used RBF as the last layer, here we'll simply use a linear layer with softmax
"""

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def buildNet():
    # LeNet-5
    model = Sequential()

    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=6, kernel_size=5,
        activation='tanh'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=16, kernel_size=5, activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28, 28, 1) / 255
    X_test = X_test.reshape(-1, 28, 28, 1) / 255

    Y_train = np_utils.to_categorical(Y_train, num_classes=10)
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)

    model = buildNet()

    print('Training...')
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

    print('Testing...')

    loss, accuracy = model.evaluate(X_test, Y_test)
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

if __name__ == '__main__':
    main()