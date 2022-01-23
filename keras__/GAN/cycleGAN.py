""" cycleGAN
    paper: Cycle-Consistent Adversarial Networks
    see: https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf
"""

from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Input, Add, Activation, Conv2DTranspose
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras__.NormalizationLayer.NormalizationLayer import InstanceNormalization
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

        if self.idx_cur+self.batch_size-1 < self.fnum:
            length = self.batch_size
            idx_nxt = self.idx_cur+self.batch_size
        else:
            length = self.fnum-self.idx_cur
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

class CycleGAN:
    def __init__(self, L_id=False):
        self.img_row = 256
        self.img_col = 256
        self.img_channels = 3
        self.img_shape = (self.img_row, self.img_col, self.img_channels)

        patch = int(self.img_row / 2 ** 4)
        self.discPatch = (patch, patch, 1)

        self.L_id = L_id
        self.buildGAN(L_id)

    def buildGenerator(self, num_resNet):
        initWeight = RandomNormal(stddev=0.02)

        def resBlock(inputs, filters):
            x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=initWeight)(inputs)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=initWeight)(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)

            x = Add()([x, inputs])
            outputs = Activation('relu')(x)
            return outputs

        def convLayer(inputs, filters, k_size=3, stride=1, act='relu'):
            x = Conv2D(filters, kernel_size=k_size, strides=stride, padding='same', kernel_initializer=initWeight)(inputs)
            x = InstanceNormalization()(x)
            outputs = Activation(act)(x)
            return outputs

        def deConvLayer(inputs, filters, k_size=3, stride=2):
            x = Conv2DTranspose(filters, kernel_size=k_size, strides=stride, padding='same', kernel_initializer=initWeight)(inputs)
            x = InstanceNormalization()(x)
            outputs = Activation('relu')(x)
            return outputs

        img_input = Input(shape=self.img_shape)

        x = convLayer(img_input, 64, k_size=7)
        x = convLayer(x, 128, stride=2)
        x = convLayer(x, 256, stride=2)

        for _ in range(num_resNet):
            x = resBlock(x, 256)

        x = deConvLayer(x, 128)
        x = deConvLayer(x, 64)
        img_output = convLayer(x, 3, k_size=7, act='tanh')

        return Model(img_input, img_output)

    def buildDiscriminator(self):
        initWeight = RandomNormal(stddev=0.02)

        def discLayer(inputs, filters, k_size=4, norm=True):
            x = Conv2D(filters, kernel_size=k_size, strides=2, padding='same', kernel_initializer=initWeight)(inputs)
            if norm:
                x = InstanceNormalization()(x)
            outputs = LeakyReLU(alpha=0.2)(x)
            return outputs

        inputImg = Input(shape=self.img_shape)

        disc1 = discLayer(inputImg, 64, norm=False)
        disc2 = discLayer(disc1, 128)
        disc3 = discLayer(disc2, 256)
        disc4 = discLayer(disc3, 512)

        validity = Conv2D(filters=1, kernel_size=4, padding='same', kernel_initializer=initWeight)(disc4)

        return Model(inputImg, validity)

    def buildGAN(self, L_id=False):
        lambda_cycle = 10.0
        lambda_id = 0.5 * lambda_cycle  # used for arts to photo mission
        optimizer = Adam(2e-4, 0.5)

        # build discriminator
        self.disc_A = self.buildDiscriminator()
        self.disc_B = self.buildDiscriminator()

        self.disc_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.disc_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # build generator, 6 ResBlock for 128x128, 9 for 256x256
        self.gen_AB = self.buildGenerator(num_resNet=9)
        self.gen_BA = self.buildGenerator(num_resNet=9)

        imgA = Input(shape=self.img_shape)
        imgB = Input(shape=self.img_shape)

        fake_B = self.gen_AB(imgA)
        fake_A = self.gen_BA(imgB)

        reconstr_A = self.gen_BA(fake_B)
        reconstr_B = self.gen_AB(fake_A)

        self.disc_A.trainable = False
        self.disc_B.trainable = False

        valid_A = self.disc_A(fake_A)
        valid_B = self.disc_B(fake_B)

        outputs = [valid_A, valid_B, reconstr_A, reconstr_B]
        loss = ['mse', 'mse', 'mae', 'mae']
        loss_weights = [1, 1, lambda_cycle, lambda_cycle]

        if L_id:
            imgA_id = self.gen_BA(imgA)
            imgB_id = self.gen_AB(imgB)

            outputs.append(imgA_id)
            loss.append('mae')
            loss_weights.append(lambda_id)

            outputs.append(imgB_id)
            loss.append('mae')
            loss_weights.append(lambda_id)

        self.combined = Model(inputs=[imgA, imgB], outputs=outputs)
        self.combined.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def trainModel(self, epochs, batch_size=1):
        self.dataLoader = DataLoader(
            'F:/wallpaper/datas/test/trainA', 'F:/wallpaper/datas/test/trainB',
            batch_size, self.img_shape
        )

        totalStep = self.dataLoader.getNumberOfBatch()
        for epoch in range(epochs):
            for step, (imgA, imgB) in enumerate(self.dataLoader):
                valid = np.ones((imgA.shape[0],) + self.discPatch)
                fake = np.zeros((imgA.shape[0],) + self.discPatch)

                fake_B = self.gen_AB.predict(imgA)
                fake_A = self.gen_BA.predict(imgB)

                discA_loss_real = self.disc_A.train_on_batch(imgA, valid)
                discA_loss_fake = self.disc_A.train_on_batch(fake_A, fake)
                D_A_loss = 0.5 * np.add(discA_loss_real, discA_loss_fake)

                discB_loss_real = self.disc_B.train_on_batch(imgB, valid)
                discB_loss_fake = self.disc_B.train_on_batch(fake_B, fake)
                D_B_loss = 0.5 * np.add(discB_loss_real, discB_loss_fake)

                D_loss = 0.5 * np.add(D_A_loss, D_B_loss)

                if self.L_id:
                    G_loss = self.combined.train_on_batch([imgA, imgB], [valid, valid, imgA, imgB, imgA, imgB])
                    print("Epoch {}/{} : Batch {}/{} -- D loss: {:.6f}, acc: {:.2f} , "
                          "G loss: {:.6f}, adv:{:.6f}, recon: {:.6f}, id: {:.6f}".format(
                        epoch+1, epochs, step+1, totalStep, D_loss[0], D_loss[1] * 100,
                                G_loss[0], np.mean(G_loss[1:3]), np.mean(G_loss[3:5]), np.mean(G_loss[5:6])
                    ))
                else:
                    G_loss = self.combined.train_on_batch([imgA, imgB], [valid, valid, imgA, imgB])
                    print("Epoch {}/{} : Batch {}/{} -- D loss: {:.6f}, acc: {:.2f} , "
                          "G loss: {:.6f}, adv:{:.6f}, recon: {:.6f}".format(
                        epoch+1, epochs, step+1, totalStep, D_loss[0], D_loss[1] * 100,
                        G_loss[0], np.mean(G_loss[1:3]), np.mean(G_loss[3:5])
                    ))

                if step % 200 == 0:
                    fpath = 'F:/wallpaper/datas/sketch/testB/1047028.png'
                    fname = 'output{}.png'.format(epoch)
                    self.colorizeImage(fpath=fpath, outputDir='output', fname=fname)

    def colorizeImage(self, fpath, outputDir, fname):
        img_input = cv2.imread(fpath, 1)
        img_input = cv2.resize(img_input, (256, 256))

        img_input = np.expand_dims(img_input, 0)
        img_input = (img_input.astype(np.float32) - 127.5) / 127.5

        img_output = self.gen_AB.predict(img_input)[0]
        img_output = img_output * 127.5 + 127.5
        img_output = img_output.astype(np.uint8)

        outputPath = os.path.join(outputDir, fname)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        cv2.imwrite(outputPath, img_output)

model = CycleGAN()
model.trainModel(epochs=5, batch_size=2)