""" pix2pix
    paper: Image-to-Image Translation with Conditional Adversarial Networks
    see: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
"""

from keras.models import Model
from keras.layers import Dropout, Conv2D, UpSampling2D, \
    LeakyReLU, Input, Concatenate, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
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

            # Normalize to [-1, 1]
            imgA[k] = (img_a.astype(np.float32) - 127.5) / 127.5
            imgB[k] = (img_b.astype(np.float32) - 127.5) / 127.5

        self.idx_cur = idx_nxt

        return imgA, imgB

class Pix2Pix:
    def __init__(self):
        self.img_row = 256
        self.img_col = 256
        self.img_channels = 3
        self.img_shape = (self.img_row, self.img_col, self.img_channels)

        patch = int(self.img_row / 2 ** 3)
        self.discPatch = (patch, patch, 1)

        self.buildGAN()

    def buildGenerator(self):
        initWeight = RandomNormal(stddev=0.02)

        def EnConv2D(inputs, filters, k_size=4, norm=True):
            x = Conv2D(filters, kernel_size=k_size, strides=2, padding='same', kernel_initializer=initWeight)(inputs)
            if norm:
                x = BatchNormalization()(x)
            outputs = LeakyReLU(alpha=0.2)(x)
            return outputs

        def DeConv2D(inputs, skipInputs, filters, k_size=4, drop_rate=0.0):
            x = UpSampling2D()(inputs)
            x = Conv2D(filters, kernel_size=k_size, padding='same', kernel_initializer=initWeight)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            outputs = Concatenate()([x, skipInputs])
            if drop_rate:
                outputs = Dropout(drop_rate)(outputs)
            return outputs

        img_input = Input(shape=self.img_shape)

        encoder1 = EnConv2D(img_input, 64, norm=False)
        encoder2 = EnConv2D(encoder1, 128)
        encoder3 = EnConv2D(encoder2, 256)
        encoder4 = EnConv2D(encoder3, 512)
        encoder5 = EnConv2D(encoder4, 512)
        encoder6 = EnConv2D(encoder5, 512)
        encoder7 = EnConv2D(encoder6, 512)

        decoder1 = DeConv2D(encoder7, encoder6, 512)
        decoder2 = DeConv2D(decoder1, encoder5, 512)
        decoder3 = DeConv2D(decoder2, encoder4, 512)
        decoder4 = DeConv2D(decoder3, encoder3, 256)
        decoder5 = DeConv2D(decoder4, encoder2, 128)
        decoder6 = DeConv2D(decoder5, encoder1, 64)

        decoder7 = UpSampling2D()(decoder6)
        img_output = Conv2D(filters=self.img_channels, kernel_size=4, padding='same', activation='tanh', kernel_initializer=initWeight)(decoder7)

        return Model(img_input, img_output)

    def buildDiscriminator(self):
        initWeight = RandomNormal(stddev=0.02)

        def discLayer(inputs, filters, k_size=4, stride=2, norm=True):
            x = Conv2D(filters, kernel_size=k_size, strides=stride, padding='same', kernel_initializer=initWeight)(inputs)
            if norm:
                x = BatchNormalization()(x)
            outputs = LeakyReLU(alpha=0.2)(x)
            return outputs

        imgA = Input(shape=self.img_shape)
        imgB = Input(shape=self.img_shape)
        inputImg = Concatenate()([imgA, imgB])

        disc1 = discLayer(inputImg, 64, norm=False)
        disc2 = discLayer(disc1, 128)
        disc3 = discLayer(disc2, 256)
        disc4 = discLayer(disc3, 512, stride=1)

        validity = Conv2D(filters=1, kernel_size=4, padding='same', activation='sigmoid', kernel_initializer=initWeight)(disc4)

        return Model([imgA, imgB], validity)

    def buildGAN(self):
        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()

        optimizer = Adam(2e-4)
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        imgB = Input(shape=self.img_shape)
        fakeA = self.generator(imgB)

        self.discriminator.trainable = False
        validity = self.discriminator([fakeA, imgB])

        self.combined = Model(imgB, [validity, fakeA])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)

    def trainModel(self, epochs, batch_size=1):
        self.dataLoader = DataLoader(
            dir_A='D:/wallpaper/datas/test/trainA',
            dir_B='D:/wallpaper/datas/test/trainB',
            batch_size=batch_size,
            img_shape=self.img_shape
        )

        totalStep = self.dataLoader.getNumberOfBatch()
        for epoch in range(epochs):
            for step, (imgA, imgB) in enumerate(self.dataLoader):
                valid = np.ones((imgA.shape[0],) + self.discPatch)
                fake = np.zeros((imgA.shape[0],) + self.discPatch)

                fakeA = self.generator.predict(imgB)

                D_loss_real = self.discriminator.train_on_batch([imgA, imgB], valid)
                D_loss_fake = self.discriminator.train_on_batch([fakeA, imgB], fake)
                D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

                G_loss = self.combined.train_on_batch(imgB, [valid, imgA])

                step += 1
                print('Epoch {}/{}, step {}/{} -- D loss: {:.4f}, acc: {:.2f}%, '
                      'G loss: {:.4f}, adver loss: {:.4f}, L1 loss: {:.4f}'.format(
                    epoch + 1, epochs, step, totalStep,  D_loss[0], D_loss[1] * 100, G_loss[0], G_loss[1], G_loss[2]))

                if step % 2 == 0:
                    fpath = 'D:/wallpaper/datas/sketch/testB/1047028.png'
                    fname = 'output{}.png'.format(epoch)
                    self.colorizeImage(fpath=fpath, outputDir='output', fname=fname)

    def colorizeImage(self, fpath, outputDir, fname):
        img_input = cv2.imread(fpath, 1)
        img_input = cv2.resize(img_input, (256, 256))

        img_input = np.expand_dims(img_input, 0)
        img_input = (img_input.astype(np.float32) - 127.5) / 127.5

        img_output = self.generator.predict(img_input)[0]
        img_output = img_output * 127.5 + 127.5
        img_output = img_output.astype(np.uint8)

        outputPath = os.path.join(outputDir, fname)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        cv2.imwrite(outputPath, img_output)

model = Pix2Pix()
model.trainModel(epochs=3, batch_size=2)