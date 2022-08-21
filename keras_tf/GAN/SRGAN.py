""" Super Resolution GAN (SRGAN)
     paper: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
     see: https://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html
"""

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Input, Activation, Add
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import Adam
# from keras.optimizers import adam_v2
from keras.applications.vgg19 import VGG19
import keras.backend as K
import numpy as np
import os
import random
import cv2

class DataLoader:
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.flist = os.listdir(dir)
        self.fnum = len(self.flist)

        self.batch_size = batch_size
        self.img_hr_shape = (256, 256, 3)
        self.img_sr_shape = (64, 64, 3)

        self.idx_cur = 0

    def getNumberOfBatch(self):
        num = self.fnum / self.batch_size + (self.fnum % self.batch_size != 0)
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

        img_hr = np.zeros((length, *self.img_hr_shape))
        img_lr = np.zeros((length, *self.img_sr_shape))

        for k in range(length):
            fpath = os.path.join(self.dir, self.flist[self.idx_cur+k])
            img_high = cv2.imread(fpath, 1)
            img_high = cv2.resize(img_high, (256, 256), interpolation=cv2.INTER_CUBIC)
            img_low = cv2.resize(img_high, (64, 64), interpolation=cv2.INTER_CUBIC)

            # Normalize to [-1, 1]
            img_hr[k] = (img_high.astype(np.float32) - 127.5) / 127.5
            img_lr[k] = (img_low.astype(np.float32) - 127.5) / 127.5

        self.idx_cur = idx_nxt

        return img_hr, img_lr

class SRGAN:
    def __init__(self):
        self.img_channels = 3
        self.input_shape_lr = (64, 64, self.img_channels)
        self.input_shape_hr = (256, 256, self.img_channels)

        self.feat_extractor = self.getFeatExtractor()

        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()

        self.discriminator.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['acc'])

        # setting trainable to false after compiling
        # the discriminator will be frozen only when training the combined
        self.discriminator.trainable = False
        self.combined = self.buildCombined(self.generator, self.discriminator)
        self.combined.compile(
            optimizer=Adam(1e-3),
            loss=['binary_crossentropy', self.perceptual_loss],
            loss_weights=[1e-3, 1]
        )

    def getFeatExtractor(self):
        vgg = VGG19(include_top=False, weights='imagenet')
        outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
        return Model(vgg.inputs, outputs_dict)

    def perceptual_loss(self, img_hr, img_sr):
        content_layer_name = 'block5_conv2'

        content_hr = self.feat_extractor(img_hr)[content_layer_name]
        content_sr = self.feat_extractor(img_sr)[content_layer_name]

        loss = K.mean(K.square(content_hr - content_sr))
        return loss

    def buildCombined(self, generator, discriminator):
        img_lr = Input(shape=self.input_shape_lr)

        img_sr = generator(img_lr)
        validity = discriminator(img_sr)

        combined = Model(img_lr, [validity, img_sr])
        return combined

    def buildGenerator(self):
        def resBlock(inputs, filters):
            x = Conv2D(filters, kernel_size=3, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(filters, kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([inputs, x])
            outputs = Activation('relu')(x)
            return outputs

        input_img = Input(shape=self.input_shape_lr)

        x = Conv2D(64, kernel_size=5, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        skip_input = x
        for _ in range(6):
            x = resBlock(x, 64)

        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([skip_input, x])
        x = Activation('relu')(x)

        x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(self.img_channels, kernel_size=5, padding='same')(x)
        output_img = Activation('tanh')(x)

        return Model(input_img, output_img)

    def buildDiscriminator(self):
        def convBlock(inputs, filters, norm=True):
            x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(inputs)
            if norm:
                x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

            x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            outputs = LeakyReLU(alpha=0.2)(x)

            return outputs

        input_img = Input(shape=self.input_shape_hr)

        x = convBlock(input_img, 64, norm=False)
        x = convBlock(x, 128)
        x = convBlock(x, 256)

        x = Flatten()(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        outputs = Dense(1, activation='sigmoid')(x)
        return Model(input_img, outputs)

    def trainModel(self, epochs, batch_size=16):
        self.dataLoader = DataLoader(
            dir='../input/imagenet/imagenet/train',
            batch_size=batch_size,
        )

        step_num = self.dataLoader.getNumberOfBatch()
        for epoch in range(epochs):
            for step, (img_hr, img_lr) in enumerate(self.dataLoader):
                num = img_hr.shape[0]
                valid = np.ones((num, 1))
                fake = np.zeros((num, 1))

                img_sr = self.generator.predict(img_lr)

                D_loss_real = self.discriminator.train_on_batch(img_hr, valid)
                D_loss_fake = self.discriminator.train_on_batch(img_sr, fake)
                D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
                
                G_loss = self.combined.train_on_batch(img_lr, [valid, img_hr])

                step += 1
                print('Epoch {}/{}, step {}/{} -- D loss: {:.4f}, acc: {:.2f}%, '
                      'G loss: {:.4f}, disc loss: {:.4f}, perceptual loss: {:.4f}'.format(
                    epoch + 1, epochs, step, step_num, D_loss[0], D_loss[1] * 100, G_loss[0], G_loss[1], G_loss[2]))

                if step % 200 == 0:
                    fpath = '../input/imagenet/imagenet/train/ILSVRC2012_val_00005012.JPEG'
                    fname = 'output{}.png'.format(epoch)
                    self.colorizeImage(fpath=fpath, outputDir='./', fname=fname)

    def colorizeImage(self, fpath, outputDir, fname):
        img_input = cv2.imread(fpath, 1)
        img_input = cv2.resize(img_input, (64, 64))

        img_input = np.expand_dims(img_input, 0)
        img_input = (img_input.astype(np.float32) - 127.5) / 127.5

        img_output = self.generator.predict(img_input)[0]
        img_output = img_output * 127.5 + 127.5
        img_output = img_output.astype(np.uint8)

        outputPath = os.path.join(outputDir, fname)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        cv2.imwrite(outputPath, img_output)

def main():
    gan = SRGAN()
    gan.trainModel(epochs=3, batch_size=8)

if __name__ == '__main__':
    main()
