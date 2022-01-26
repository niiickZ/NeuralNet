""" getting started to keras and RNN
    IMDB classification
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras.preprocessing.sequence import pad_sequences

def getData(mxfeatures, mxlen):
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=mxfeatures)  # 只要出现频率前10000的单词

    X_train = pad_sequences(X_train, maxlen=mxlen) # 将句子统一长度
    X_test = pad_sequences(X_test, maxlen=mxlen)

    return (X_train, Y_train), (X_test, Y_test)

def buildNet(mxfeatures, input_length):
    model = Sequential()
    model.add(Embedding(mxfeatures, 32, input_length=input_length))

    model.add(SimpleRNN(16))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

mxfeatures = 10000
mxlen = 600
(X_train, Y_train), (X_test, Y_test) = getData(mxfeatures, mxlen)

model = buildNet(mxfeatures, mxlen)
model.summary()
# model.fit(X_train, Y_train, epochs=3, batch_size=128, validation_split=0.2)