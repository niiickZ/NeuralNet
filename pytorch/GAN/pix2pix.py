""" pix2pix
    paper: Image-to-Image Translation with Conditional Adversarial Networks
    see: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
"""


import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import os
import cv2
import numpy as np
import random


class DataLoader:
    """ dataloader for image to image translation model """
    def __init__(self, dir_A, dir_B, batch_size, img_shape):
        """
        :param dir_A: directory of input images
        :param dir_B: directory of output images (paired images should have same file name)
        :param batch_size: mini-batch size
        :param img_shape: shape of the image
        """
        self.dir_A = dir_A
        self.dir_B = dir_B

        self.flist = os.listdir(dir_A)
        self.fnum = len(self.flist)

        self.batch_size = batch_size
        self.img_shape = img_shape

        self.idx_cur = 0

    def getNumberOfBatch(self):
        num = self.fnum / self.batch_size
        if self.fnum % self.batch_size != 0:
            num += 1
        return int(num)

    def reset(self):
        self.idx_cur = 0
        random.shuffle(self.flist)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx_cur >= self.fnum:
            self.reset()
            raise StopIteration

        if self.idx_cur + self.batch_size - 1 < self.fnum:
            length = self.batch_size
            idx_nxt = self.idx_cur + self.batch_size
        else:
            length = self.fnum - self.idx_cur
            idx_nxt = self.fnum

        imgA = np.zeros((length, *self.img_shape))
        imgB = np.zeros((length, *self.img_shape))

        for k in range(length):
            fpath_A = os.path.join(self.dir_A, self.flist[self.idx_cur+k])
            fpath_B = os.path.join(self.dir_B, self.flist[self.idx_cur+k])

            img_a = cv2.imread(fpath_A, 1)
            img_b = cv2.imread(fpath_B, 1)

            # resize images
            img_a = cv2.resize(img_a, (self.img_shape[1], self.img_shape[2]))
            img_b = cv2.resize(img_b, (self.img_shape[1], self.img_shape[2]))

            # we need "channel-first" image for pytorch
            img_a = np.transpose(img_a, (2, 0, 1))
            img_b = np.transpose(img_b, (2, 0, 1))

            # Normalize to [-1, 1]
            imgA[k] = (img_a.astype(np.float32) - 127.5) / 127.5
            imgB[k] = (img_b.astype(np.float32) - 127.5) / 127.5

        self.idx_cur = idx_nxt

        imgA = torch.from_numpy(imgA).type(torch.float32)
        imgB = torch.from_numpy(imgB).type(torch.float32)
        return imgA, imgB

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, drop_rate=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2))

        if drop_rate:
            layers.append(nn.Dropout2d(drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dropout = None
        if drop_rate:
            self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, inputs, skip_inputs):
        x = self.model(inputs)
        x = torch.cat((x, skip_inputs), dim=1)
        if self.dropout:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # the first encoder block doesn't has normalization layer
        self.encoder1 = UNetEncoder(in_channels, 64, norm=False)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)
        self.encoder5 = UNetEncoder(512, 512)
        self.encoder6 = UNetEncoder(512, 512)
        self.encoder7 = UNetEncoder(512, 512)

        self.decoder1 = UNetDecoder(512, 512, drop_rate=0.5)
        self.decoder2 = UNetDecoder(1024, 512, drop_rate=0.5)
        self.decoder3 = UNetDecoder(1024, 512, drop_rate=0.5)
        self.decoder4 = UNetDecoder(1024, 256, drop_rate=0.5)
        self.decoder5 = UNetDecoder(512, 128, drop_rate=0.5)
        self.decoder6 = UNetDecoder(256, 64, drop_rate=0.5)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )

    def forward(self, input_img):
        en1 = self.encoder1(input_img)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        en6 = self.encoder6(en5)
        en7 = self.encoder7(en6)

        de1 = self.decoder1(en7, en6)
        de2 = self.decoder2(de1, en5)
        de3 = self.decoder3(de2, en4)
        de4 = self.decoder4(de3, en3)
        de5 = self.decoder5(de4, en2)
        de6 = self.decoder6(de5, en1)

        output_img = self.final(de6)
        return output_img

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.disc1 = self.discLayer(in_channels * 2, 64, norm=False)
        self.disc2 = self.discLayer(64, 128)
        self.disc3 = self.discLayer(128, 256)
        self.disc4 = self.discLayer(256, 512)

        self.final = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
            nn.Sigmoid()
        )

    def discLayer(self, in_channels, out_channels, stride=2, norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def forward(self, input_imgA, input_imgB):
        inputs = torch.cat((input_imgA, input_imgB), dim=1)
        x = self.disc1(inputs)
        x = self.disc2(x)
        x = self.disc3(x)
        x = self.disc4(x)
        validity = self.final(x)
        return validity

class Pix2Pix():
    def __init__(self):
        self.cuda_on = torch.cuda.is_available()

        self.img_shape = (3, 256, 256)
        self.in_channels = 3
        self.out_channels = 3

        self.patch = (1, self.img_shape[1] // 2**4, self.img_shape[2] // 2**4)

        self.generator = Generator(self.in_channels, self.out_channels)
        self.discriminator = Discriminator(self.in_channels)

        self.optim_G = Adam(self.generator.parameters(), lr=2e-4)
        self.optim_D = Adam(self.discriminator.parameters(), lr=2e-4)

        self.loss_adver = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()
        self.lambda_l1 = 100

        if self.cuda_on:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss_adver.cuda()

    def train(self, epochs=1, batch_size=32):
        dataLoader = DataLoader(
            dir_A='../input/image-colorization-dataset/data/train_color',
            dir_B='../input/image-colorization-dataset/data/train_black',
            batch_size=batch_size,
            img_shape=self.img_shape
        )

        for epoch in range(epochs):
            for step, (img_A, img_B) in enumerate(dataLoader):

                valid = torch.ones((img_A.shape[0], ) + self.patch, dtype=torch.float32)
                fake = torch.zeros((img_A.shape[0], ) + self.patch, dtype=torch.float32)

                if self.cuda_on:
                    valid = valid.cuda()
                    fake = fake.cuda()
                    img_A = img_A.cuda()
                    img_B = img_B.cuda()

                # generate fake images
                fake_A = self.generator(img_B)

                # Train Discriminator
                D_loss_real = self.loss_adver(self.discriminator(img_A, img_B), valid)
                D_loss_fake = self.loss_adver(self.discriminator(fake_A, img_B), fake)
                D_loss = (D_loss_real + D_loss_fake) / 2

                self.optim_D.zero_grad()
                D_loss.backward(retain_graph=True)  # retain_graph=True: retain the computational graph
                self.optim_D.step()

                # Train Generator
                loss_adver = self.loss_adver(self.discriminator(fake_A, img_B), valid)
                loss_l1 = self.loss_l1(fake_A, img_A)
                G_loss = loss_adver + self.lambda_l1 * loss_l1

                self.optim_G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

                print('Epoch: {}, Step: {}, D_loss: {:.5f}, adver loss: {:.5f}, l1 loss: {:.5f}'.format(
                    epoch+1, step, D_loss.item(), loss_adver.item(), loss_l1.item()))

                if (step+1) % 400 == 0:
                    torchvision.utils.save_image(
                        fake_A.detach()[0:1], './{}_{}.png'.format(epoch, step))

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=10, batch_size=4)