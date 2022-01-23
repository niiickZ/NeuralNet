""" Deep Convolutional GANs
    paper: Unsupervised Representation Learning with Deep Convolution Generative Adversarial Networks
    see: https://arxiv.org/pdf/1511.06434.pdf
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.dense = nn.Sequential(
            nn.Linear(self.input_shape, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7, momentum=0.8),
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, tensor_input):
        x = self.dense(tensor_input)
        x = x.reshape(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.4),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.4),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.dense(x)

        return output

class DCGAN():
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
                z = torch.randn(num, self.input_shape)

                if self.cuda_on:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    z = z.cuda()
                    img_real = img_real.cuda()

                img_gen = self.generator(z)

                # Train Discriminator
                D_loss_real = self.loss_adver(self.discriminator(img_real), valid)
                D_loss_fake = self.loss_adver(self.discriminator(img_gen), fake)
                D_loss = (D_loss_real + D_loss_fake) / 2

                self.optim_D.zero_grad()
                D_loss.backward(retain_graph=True)
                self.optim_D.step()

                # Train Generator
                G_loss = self.loss_adver(self.discriminator(img_gen), valid)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                print('Epoch:', epoch+1, ' Step:', step, ' D_loss:', D_loss.item(), ' G_loss:', G_loss.item())

                if (step+1) % 400 == 0:
                    torchvision.utils.save_image(
                        img_gen.data[:9], 'output\\{}_{}.png'.format(epoch, step), nrow=3)

if __name__ == '__main__':
    gan = DCGAN()
    gan.train(epochs=10, batch_size=64)