""" Conditional GAN
    paper: Conditional Generative Adversarial Nets
    see: https://arxiv.org/pdf/1411.1784.pdf
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import numpy as np


class Generator(nn.Module):
    """define the generator"""
    def __init__(self, latent_dim, output_shape, num_classes):
        super().__init__()

        self.output_shape = output_shape

        self.embedding = nn.Embedding(num_classes, latent_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, 256),
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

    def forward(self, inputs, label):
        label_embed = self.embedding(label)
        label_embed = label_embed.view(label_embed.shape[0], -1)

        x = torch.mul(inputs, label_embed)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output_img = x.reshape(-1, *self.output_shape)

        return output_img


class Discriminator(nn.Module):
    """define the discriminator"""
    def __init__(self, input_shape, num_classes):
        super().__init__()

        embedding_dim = int(np.prod(input_shape))
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(64, 1)
        )

    def forward(self, input_img, label):
        label_embed = self.embedding(label)
        label_embed = label_embed.view(label_embed.shape[0], -1)

        img_flat = input_img.view(input_img.shape[0], -1)
        x = torch.mul(label_embed, img_flat)

        outputs = self.model(x)
        return outputs

class CGAN():
    def __init__(self):
        self.cuda_on = torch.cuda.is_available()

        self.latent_dim = 100
        self.img_shape = (1, 28, 28)
        self.num_classes = 10

        self.generator = Generator(self.latent_dim, self.img_shape, self.num_classes)
        self.discriminator = Discriminator(self.img_shape, self.num_classes)

        self.optim_G = Adam(self.generator.parameters(), lr=2e-4)
        self.optim_D = Adam(self.discriminator.parameters(), lr=2e-4)
        self.loss_adver = nn.BCEWithLogitsLoss()

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
        loader = self.getDataloader('D:/wallpaper/datas/pytorch/', batch_size)

        for epoch in range(epochs):
            for step, (img_real, label) in enumerate(loader):
                label = label.view(-1, 1)
                num = img_real.shape[0]

                valid = torch.ones((num, 1), dtype=torch.float32)
                fake = torch.zeros((num, 1), dtype=torch.float32)

                # standard normal distribution
                z = torch.randn(num, self.latent_dim)
                label_gen = torch.randint(0, 10, (num, 1))

                if self.cuda_on:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    z = z.cuda()
                    label_gen = label_gen.cuda()
                    img_real = img_real.cuda()
                    label = label.cuda()

                # generate fake images
                img_gen = self.generator(z, label_gen)

                # train the discriminator
                D_loss_real = self.loss_adver(self.discriminator(img_real, label), valid)
                D_loss_fake = self.loss_adver(self.discriminator(img_gen, label_gen), fake)
                D_loss = (D_loss_real + D_loss_fake) / 2

                self.optim_D.zero_grad()
                D_loss.backward(retain_graph=True)  # retain_graph=True: retain the computational graph
                self.optim_D.step()

                # train the generator
                G_loss = self.loss_adver(self.discriminator(img_gen, label_gen), valid)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                print('Epoch: {}, Step: {}, D_loss: {:.5f}, G_loss: {:.5f}'.format(
                    epoch+1, step, D_loss.item(), G_loss.item()))

                if (step+1) % 400 == 0:
                    fpath = 'output\\{}_{}.png'.format(epoch, step)
                    self.generateImage(fpath)


    def generateImage(self, fpath):
        r, c = 3, 10
        z = torch.randn(r * c, self.latent_dim)
        label = torch.tile(torch.arange(0, 10, dtype=torch.int32), (r, )).view(-1, 1)

        if self.cuda_on:
            z = z.cuda()
            label = label.cuda()

        img_gen = self.generator(z, label)
        torchvision.utils.save_image(img_gen.detach(), fpath, nrow=10)

if __name__ == '__main__':
    gan = CGAN()
    gan.train(epochs=10, batch_size=64)