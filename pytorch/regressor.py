""" getting started to Neural Net and pytorch
    example of linear regression
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD

def createData():
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

    X = np.expand_dims(X, 1)
    Y = np.expand_dims(Y, 1)

    return X, Y

'''定义网络'''
class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        '''定义网络需要的层'''
        self.layer = nn.Linear(1, 1)  # 线性全连接层，Linear(in_features, out_features, bias=True)

    '''定义前向传播'''
    def forward(self, input):
        output = self.layer(input)
        return output

X, Y = createData()
X_ten = torch.from_numpy(X).to(torch.float32)
Y_ten = torch.from_numpy(Y).to(torch.float32)  # Linear只支持float32

regressor = Regressor() # 实例化网络

optimizer = SGD(regressor.parameters(), lr=0.2)  # 定义优化器
loss_func = nn.MSELoss()  # 定义误差计算公式

'''训练'''
for _ in range(100):
    pred = regressor(X_ten)     # 预测

    loss = loss_func(pred, Y_ten)     # 计算误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 反向传播, 计算梯度
    optimizer.step()        # 将梯度更新应用到optimizer对应的参数上

'''获取训练后的网络参数'''
# .state_dict()返回各层参数的字典
# >>> OrderedDict([('layer.weight', tensor([[0.4949]])), ('layer.bias', tensor([2.0030]))])

w = regressor.state_dict()['layer.weight']
b = regressor.state_dict()['layer.bias']

Y_pred = w * X + b
# Y_pred = regressor(X_ten)
# Y_pred = Y_pred.detach().numpy()

'''模型保存'''
# torch.save(regressor, 'net.pkl')  # 保存整个网络
# torch.save(regressor.state_dict(), 'net_params.pkl')  # 只保存参数

# regressor = torch.load('net.pkl')  # 读取整个网络
# regressor.load_state_dict('net_params.pkl')  # 只读取参数

# 结果可视化
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
