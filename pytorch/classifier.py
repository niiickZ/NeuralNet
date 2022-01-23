""" getting started to Neural Net and pytorch
    Mnist hand-written digital number classification
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
import numpy as np

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    '''定义前向传播'''

    def forward(self, input):
        output = self.model(input)
        return output

def getData(datadir):
    data_train = torchvision.datasets.MNIST(
        root=datadir,  # 数据保存位置
        train=True,  # 使用训练数据or测试数据
        transform=torchvision.transforms.ToTensor(),  # 将PIL或numpy转换为Tensor,送入dataloader后自动归一化到[0,1]
        download=True,  # 是否下载(若已下载则忽略)
    )

    data_test = torchvision.datasets.MNIST(root=datadir, train=False)

    return data_train, data_test


cuda_on = torch.cuda.is_available()

classifier = Classifier()
optimizer = Adam(classifier.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()  # 使用交叉熵损失时会自动将label转化为one-hot编码,同时也自动对输入求softmax

if cuda_on:  # 开启GPU模式
    classifier.cuda()
    loss_func.cuda()

data_train, data_test = getData('F:/wallpaper/datas/pytorch/')
loader = Data.DataLoader(dataset=data_train, batch_size=32, shuffle=True)

for epoch in range(1):
    for step, (X, Y) in enumerate(loader):
        if cuda_on:  # 开启GPU模式
            X = X.cuda()
            Y = Y.cuda()

        pred = classifier(X)
        loss = loss_func(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch:', epoch+1, ' Step:', step, ' loss:', loss.item())

'''测试'''
test_X = torch.unsqueeze(data_test.data, dim=1).type(torch.float32) / 255.
test_Y = data_test.targets

if cuda_on:  # 开启GPU模式
    test_X = test_X.cuda()
    test_Y = test_Y.cuda()

output = classifier(test_X)
label_pred = torch.argmax(output, dim=1)  # 按行求最大值索引

# print(test_Y)
# print(label_pred)
print('accuracy:', np.sum((test_Y==label_pred).tolist()) / test_Y.shape[0])
