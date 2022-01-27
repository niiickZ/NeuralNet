""" Least Square GAN
    paper: Least Squares Generative Adversarial Networks
    see: https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf
"""


import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
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

    def forward(self, inputs):
        x = self.dense(inputs)
        x = x.reshape(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output_img = self.conv4(x)
        return output_img

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=2, padding=1),
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
            nn.Linear(256 * 7 * 7, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, input_img):
        x = self.conv1(input_img)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.dense(x)

        return output

class LSGAN():
    def __init__(self):
        self.cuda_on = torch.cuda.is_available()

        self.latent_dim = 100
        self.img_shape = (1, 28, 28)

        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator(self.img_shape)

        self.optim_G = Adam(self.generator.parameters(), lr=2e-4)
        self.optim_D = Adam(self.discriminator.parameters(), lr=2e-4)
        self.loss_adver = nn.MSELoss()

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
            for step, (img_real, _) in enumerate(loader):
                num = img_real.shape[0]

                valid = torch.ones((num, 1), dtype=torch.float32)
                fake = torch.zeros((num, 1), dtype=torch.float32)

                # standard normal distribution
                z = torch.randn(num, self.latent_dim)

                if self.cuda_on:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    z = z.cuda()
                    img_real = img_real.cuda()

                # generate fake images
                img_gen = self.generator(z)

                # Train Discriminator
                D_loss_real = self.loss_adver(self.discriminator(img_real), valid)
                D_loss_fake = self.loss_adver(self.discriminator(img_gen), fake)
                D_loss = (D_loss_real + D_loss_fake) / 2

                self.optim_D.zero_grad()
                D_loss.backward(retain_graph=True)  # retain_graph=True: retain the computational graph
                self.optim_D.step()

                # Train Generator
                G_loss = self.loss_adver(self.discriminator(img_gen), valid)

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                print('Epoch: {}, Step: {}, D_loss: {:.5f}, G_loss: {:.5f}'.format(
                    epoch+1, step, D_loss.item(), G_loss.item()))

                if (step+1) % 400 == 0:
                    torchvision.utils.save_image(
                        img_gen.detach()[:9], 'output\\{}_{}.png'.format(epoch, step), nrow=3)

if __name__ == '__main__':
    gan = LSGAN()
    gan.train(epochs=10, batch_size=64)