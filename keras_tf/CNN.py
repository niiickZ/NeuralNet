""" getting started to keras and CNN_classifier
    Mnist hand-written digital number classification
"""

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

def buildNet():
    model = Sequential()

    # output_shape(28, 28, 32)
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=32,  # 滤波器个数
        kernel_size=(5, 5),  # 卷积核大小
        # strides=(1, 1), 卷积沿宽度和高度方向的步长,默认(1,1)
        padding='same',  # 边缘填充方法,可选"valid"/"same","valid"-不填充,"same" 填充以使图片维度不变
        data_format='channels_last',  # 指定通道在哪一维度
        # dilation_rate=(1,1), 膨胀卷积的膨胀率
        # use_bias=True, 该层是否使用偏置向量
        # activation, 激活函数
    ))
    model.add(Activation('relu'))

    # output_shape(14, 14, 32)
    model.add(MaxPooling2D(
        pool_size=(2, 2),  # 2x2窗口的最大池化
        strides=(2, 2),  # 步长,如果是None则使用pool_size
        padding='same',
    ))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))  # output_shape(14, 14, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # output_shape(7, 7, 64)

    model.add(Flatten())  # 一维展开，无学习参数

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25)) # 引入dropout防止过拟合,rate表示丢弃的比例

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model

def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # (60000, 28, 28, 1),第四维代表channel
    X_train = X_train.reshape(-1, 28, 28, 1) / 255
    X_test = X_test.reshape(-1, 28, 28, 1) / 255

    Y_train = np_utils.to_categorical(Y_train, num_classes=10)
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)

    model = buildNet()

    print('Training...')
    model.fit(X_train, Y_train, epochs=5, batch_size=64, verbose=1)

    print('Testing...')

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

if __name__ == '__main__':
    main()