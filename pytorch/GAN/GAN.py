""" original GAN
     paper: Generative Adversarial Nets
     see: https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import numpy as np

"""define the generator"""
class Generator(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_shape, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(1024, int(np.prod(self.output_shape))),
            nn.Sigmoid()
        )

    def forward(self, tensor_input):
        x = self.layer1(tensor_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = x.reshape(-1, *self.output_shape)

        return output


"""define the discriminator"""
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(int(np.prod(self.input_shape)), 512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        output = self.model(img)

        return output

class GAN():
    def __init__(self):
        self.cuda_on = torch.cuda.is_available()

        self.input_shape = 100
        self.img_shape = (1, 28, 28)

        self.generator = Generator(self.input_shape, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

        self.optim_G = Adam(self.generator.parameters(), lr=2e-4)
        self.optim_D = Adam(self.discriminator.parameters(), lr=2e-4)
        self.loss_adver = nn.BCELoss()

        if self.cuda_on:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss_adver.cuda()

    def getDataloader(self, dataDir, batch_size):
        mnist = torchvision.datasets.MNIST(
            root=dataDir, train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        loader = Data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
        return loader

    def train(self, epochs=1, batch_size=32):
        loader = self.getDataloader('F:/wallpaper/datas/pytorch/', batch_size)

        for epoch in range(epochs):
            for step, (img_real, _) in enumerate(loader):
                num = img_real.shape[0]

                valid = torch.ones((num, 1), dtype=torch.float32)
                fake = torch.zeros((num, 1), dtype=torch.float32)

                # standard normal distribution
                z = torch.randn(num, self.input_shape)

                if self.cuda_on:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    z = z.cuda()
                    img_real = img_real.cuda()

                # generator以输入的随机噪声生成假图片
                img_gen = self.generator(z)

                # train the discriminator
                D_loss_real = self.loss_adver(self.discriminator(img_real), valid)
                D_loss_fake = self.loss_adver(self.discriminator(img_gen), fake)
                D_loss = (D_loss_real + D_loss_fake) / 2

                self.optim_D.zero_grad()
                D_loss.backward(retain_graph=True)  # retain_graph=True 保留计算图
                self.optim_D.step()

                # train the generator
                G_loss = self.loss_adver(self.discriminator(img_gen), valid)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                print('Epoch:', epoch+1, ' Step:', step, ' D_loss:', D_loss.item(), ' G_loss:', G_loss.item())

                if (step+1) % 400 == 0:
                    torchvision.utils.save_image(
                        img_gen.data[:9], 'output\\{}_{}.png'.format(epoch, step), nrow=3)

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=12, batch_size=64)