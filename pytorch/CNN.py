""" getting started to pytorch and CNN
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

        '''
        def __init__(self,
             in_channels: int,
             out_channels: int,
             kernel_size: Union[int, tuple],
             stride: Any = 1,
             padding: Any = 0,
             dilation: Any = 1,
             groups: int = 1,
             bias: bool = True,
             padding_mode: str = 'zeros') -> None
        '''
        # pytorch中张量维度顺序为(batch_size, channel, H, W)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.layer = nn.Linear(7 * 7 * 32, 256)
        self.outputLayer = nn.Linear(256, 10)

    def forward(self, tensor_input):
        x = self.conv1(tensor_input)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)  # 一维展开,展开为后(batch_size, channel*H*W)
        x = self.layer(x)
        out = self.outputLayer(x)

        return out

def getData(datadir):
    data_train = torchvision.datasets.MNIST(
        root=datadir,  # 数组保存位置
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
test_X = torch.unsqueeze(data_test.data, dim=1).type(torch.float32) / 255.  # 将torchvision.datasets.MNIST对象转换为Tensor对象
test_Y = data_test.targets

if cuda_on:  # 开启GPU模式
    test_X = test_X.cuda()
    test_Y = test_Y.cuda()

output = classifier(test_X)
label_pred = torch.argmax(output, dim=1)  # 按行求最大值索引

# print(test_Y)
# print(label_pred)
print('accuracy:', np.sum((test_Y==label_pred).tolist()) / test_Y.shape[0])
