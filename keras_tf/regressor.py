""" getting started to Neural Net and keras
    example of linear regression
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import plot_model
import matplotlib.pyplot as plt

def createData():
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

    X = np.expand_dims(X, 1)  # 使X.shape为(200, 1)，表示200个一维数据
    Y = np.expand_dims(Y, 1)

    X_Train, Y_Train = X[:160], Y[:160]
    X_Test, Y_Test = X[160:], Y[160:]

    return X_Train, Y_Train, X_Test, Y_Test

X_Train, Y_Train, X_Test, Y_Test = createData()

# plt.scatter(X_Train, Y_Train)
# plt.show()

# 建立神经网络
model = Sequential()
model.add(Dense(units=1, input_shape=(1,))) # units即旧版本output_dim

# 损失函数和优化器
model.compile(loss='mse', optimizer='sgd')

print(model.summary())

print("Training...")
hist = model.fit(X_Train, Y_Train, batch_size=40, epochs=200, verbose=1)

print(hist.history['loss'])

# another way to train
# for step in range(301):
#     cost = model.train_on_batch(X_Train, Y_Train)

print("Testing...")
cost = model.evaluate(X_Test, Y_Test, batch_size=40)
print("Testing Cost: ",cost)

k, b = model.layers[0].get_weights()
print("k: {}, b: {}".format(k,b))

Y_Pred = model.predict(X_Test)
plt.scatter(X_Test, Y_Test)
plt.plot(X_Test, Y_Pred, 'red')
plt.show()

# plot_model(model, to_file='regression.png', show_shapes=True)