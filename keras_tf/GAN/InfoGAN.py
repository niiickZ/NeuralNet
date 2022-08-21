""" Information Maximizing GAN
    paper: InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
    see: https://proceedings.neurips.cc/paper/2016/hash/7c9d0b1f96aebd7b5eca8c3edaa19ebb-Abstract.html
"""

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, \
    LeakyReLU, Conv2D, Conv2DTranspose, Activation, Dropout
from keras.optimizers import Adam
# from keras.optimizers import adam_v2
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class InfoGAN:
    def __init__(self):
        self.img_row = 28
        self.img_col = 28
        self.channel = 1
        self.img_shape = (self.img_row, self.img_col, self.channel)

        self.noise_dim = 62
        self.num_class = 10
        self.num_latent_code = 1
        self.latent_dim = self.noise_dim + self.num_class + self.num_latent_code

        self.buildGAN()

    def buildGenerator(self):
        inputs = Input(shape=(self.latent_dim,))

        x = Dense(1024)(inputs)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Dense(7 * 7 * 256)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Reshape((7, 7, 256))(x)

        x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        img = Conv2DTranspose(self.channel, kernel_size=3, padding='same', activation='tanh')(x)

        return Model(inputs, img)

    def buildDiscriminator(self):
        input_img = Input(shape=self.img_shape)

        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(input_img)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)

        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)

        x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)

        validity = Dense(1, activation='sigmoid')(x)
        label = Dense(self.num_class, activation='softmax')(x)

        c2 = Dense(2)(x)
        c2 = Activation(self.dev_positive)(c2)

        return Model(input_img, validity), Model(input_img, [label, c2])

    def dev_positive(self, x):
        # the standard deviation is parameterized through an exponential transformation to ensure positivity
        x = K.concatenate([x[:, 0:1], K.exp(x[:, 1:2])])
        return x

    def gauss_kl_divergence(self, p, q):
        # KL divergence between two gaussian distributions
        mu1, sigma1 = p[:, 0], p[:, 1]
        mu2, sigma2 = q[:, 0], q[:, 1]
        t1 = K.log(sigma2 / sigma1)
        t2 = (K.square(sigma1) + K.square(mu1 - mu2)) / K.square(sigma2) / 2
        kl_div = K.mean(t1 + t2 + 0.5)
        return kl_div

    def mutual_info_loss(self, c, c_given_x):
        # LI(G, Q) given by the paper
        eps = 1e-9
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        loss = conditional_entropy + entropy
        return loss

    def buildGAN(self):
        self.generator = self.buildGenerator()
        self.discriminator, self.auxiliary = self.buildDiscriminator()
        self.discriminator.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['acc'])

        inputs = Input(shape=(self.latent_dim,))
        img = self.generator(inputs)

        # setting trainable to false after compiling
        # the discriminator will be frozen only when training the combined
        self.discriminator.trainable = False

        validity = self.discriminator(img)
        label, c2 = self.auxiliary(img)

        self.combined = Model(inputs, [validity, label, c2])

        # for latent code of labels, LI(G, Q) is equivalent to cross entropy
        self.combined.compile(
            optimizer=Adam(1e-3),
            loss=['binary_crossentropy', 'categorical_crossentropy', self.gauss_kl_divergence]
        )

    def getRandomInput(self, batch_size):
        # noise with standard normal distribution,
        noise = np.random.normal(0, 1, (batch_size, self.noise_dim))

        # randomly generated category label (discrete latent code)
        labels = np.random.randint(0, self.num_class, (batch_size,))
        labels = np_utils.to_categorical(labels, num_classes=self.num_class)

        # continuous latent code c2
        latent_c2 = np.random.normal(0, 1 / 3, (batch_size, 1))

        # concatenate noise and latent code
        inputs = np.hstack((noise, labels, latent_c2))
        assert inputs.shape == (batch_size, self.latent_dim)

        return inputs, labels

    def trainModel(self, epochs, batch_size=64):
        (X_train, Y_train), (_, _) = mnist.load_data()

        # Normalize to [-1, 1]
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            epoch += 1

            # randomly select a batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            org_imgs = X_train[idx]

            # generate fake images
            inputs, labels = self.getRandomInput(batch_size)
            genImgs = self.generator.predict(inputs)

            # train the discriminator
            D_loss_real = self.discriminator.train_on_batch(org_imgs, valid)
            D_loss_fake = self.discriminator.train_on_batch(genImgs, fake)
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

            # train the generator
            c2 = np.zeros((batch_size, 2))
            c2[:, 1] = 1 / 3
            G_loss = self.combined.train_on_batch(inputs, [valid, labels, c2])

            print("epoch {} --- D loss: {:.4f}, acc: {:.2f}, G loss: {:.4f}".format(
                epoch, D_loss[0], 100 * D_loss[1], G_loss[0]))

            if epoch % 1000 == 0:
                self.saveImage(epoch)

    def saveImage(self, epoch):
        r, c = 3, 10

        noise = np.random.normal(0, 1, (r * c, self.noise_dim))

        # [[0, 1, 2, ..., 9], [0, 1, 2, ..., 9], [0, 1, 2, ..., 9]]
        labels = np.tile(np.arange(0, c), r)
        labels = np_utils.to_categorical(labels, num_classes=self.num_class)

        # [[-1, -1, ..., -1], [0, 0, ..., 0], [1, 1, ..., 1]]
        c2 = np.zeros((r * c, 1))
        c2[:c, :], c2[r * c - c + 1:, :] = -1, 1

        inputs = np.hstack((noise, labels, c2))
        assert inputs.shape == (r * c, self.latent_dim)
        genImgs = self.generator.predict(inputs)

        # Rescale to [0, 1]
        genImgs = 0.5 * genImgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(genImgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('output/%d.png' % epoch)
        plt.close()


def main():
    infoGAN = InfoGAN()
    infoGAN.trainModel(epochs=20000, batch_size=64)


if __name__ == '__main__':
    main()
