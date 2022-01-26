""" ResNet 34-layer
    paper: Deep Residual Learning for Image Recognition
    see: https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
"""

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channel, filters, downsample=False):
        super().__init__()

        self.downsample = downsample
        stride = 2 if downsample else 1

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, filters, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters)
        )

        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, filters, kernel_size=1, stride=2, padding=0),
                nn.ReLU(),
            )

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        if self.downsample:
            inputs = self.shortcut(inputs)

        outputs = torch.add(inputs, x)
        return outputs

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channel = 3
        self.buildNet()

    def buildNet(self):
        def buildBlock(num, input_channel, filters, downsample=True):
            layer = []
            for idx in range(num):
                downsample = (idx == 0 and downsample)
                input_channel = filters if idx !=0 else input_channel
                layer.append(ResBlock(input_channel=input_channel, filters=filters, downsample=downsample))

            return nn.Sequential(*layer)

        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        num_block = [3, 4, 6, 3]

        self.conv2 = buildBlock(num_block[0], input_channel=64, filters=64, downsample=False)
        self.conv3 = buildBlock(num_block[1], input_channel=64, filters=128)
        self.conv4 = buildBlock(num_block[2], input_channel=128, filters=256)
        self.conv5 = buildBlock(num_block[3], input_channel=256, filters=512)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))  # global average pooling
        self.classifier = nn.Linear(512, 10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        outputs = self.classifier(x)
        return outputs

resnet = ResNet()