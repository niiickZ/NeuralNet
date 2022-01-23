""" getting started to Neural Net and keras
    Mnist hand-written digital number classification
"""

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# 下载mnist数据集到 '~/.keras/datasets/'
# X_train shape (60000, 28, 28), Y_train shape (60000, )
# X_test shape (10000, 28, 28), Y_test shape (10000, )

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 或直接从本地读取
# datas = np.load('datas\\mnist.npz')
# X_train = datas['x_train']
# Y_train = datas['y_train']
# X_test = datas['x_test']
# Y_test = datas['y_test']

# 将每个图片矩阵一维序列化并将数值归一化
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255

# 将整型标签转化为onehot编码，num_classes为标签类别数
# 例如5转化为[0, 0, 0, 0, 0, 5, 0, 0, 0, 0](编码从0开始)
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# 建立神经网络

# method-1
# model = Sequential([
#     Dense(32, input_dim=784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])

# method-2
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(lr=0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training...')
# train the model
# epochs=1:对整个数据集的训练次数 / batch_size=32:将数据集划分为训练集时每个集合样本数
# verbose=1:训练日志 0-不显示 / 1-每个epoch显示进度条 / 2-每个epoch显示loss和accuracy
# validation_split=0.: 用于验证集的数据比例
# validation_data: 形式为(X,Y)的元组,指定的验证集,优先于validation_split
# shuffle:是否在每轮迭代之前混洗数据
model.fit(X_train, Y_train, epochs=2, batch_size=32, verbose=1)

print('Testing...')
loss, accuracy = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)

print('test loss: ', loss)
print('test accuracy: ', accuracy)