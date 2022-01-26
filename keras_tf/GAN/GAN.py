""" original GAN
     paper: Generative Adversarial Nets
     see: https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
"""

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, Activation
from keras.layers import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class GAN:
    def __init__(self):
        self.img_shape = (28, 28, 1)
        self.latent_dim = 100

        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()

        input = Input(shape=(self.latent_dim,))
        img = self.generator(input)

        # setting trainable to false after compiling
        # the discriminator will be frozen only when training the combined
        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(input, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=Adam(2e-4)
        )

    def buildGenerator(self):
        model = Sequential()

        model.add(Dense(input_dim=self.latent_dim, units=256))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def buildDiscriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(0.2))

        model.add(Dense(64))
        model.add(LeakyReLU(0.2))

        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        discriminator = Model(img, validity)
        discriminator.compile(
            optimizer=Adam(2e-4),
            loss='binary_crossentropy',
        )
        return discriminator

    def trainModel(self, epochs, batch_size=64):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        # Normalize to [-1, 1]
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # target tensor
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            epoch += 1

            # randomly select a batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            orgImg = X_train[idx]

            # standard normal distribution
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # generate fake images
            genImg = self.generator.predict(noise)

            # train the discriminator
            D_loss_real = self.discriminator.train_on_batch(orgImg, valid)
            D_loss_fake = self.discriminator.train_on_batch(genImg, fake)
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

            # train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            G_loss = self.combined.train_on_batch(noise, valid)

            print("{} --- D loss: {:.4f} , G loss: {:.4f}".format(epoch, D_loss, G_loss))

            if epoch % 200 == 0:
                self.saveImage(epoch)

    def saveImage(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        genImgs = self.generator.predict(noise)

        # Rescale to [0, 1]
        genImgs = 0.5 * genImgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(genImgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('output\\%d.png' % epoch)
        plt.close()

def main():
    gan = GAN()
    gan.trainModel(epochs=8000, batch_size=64)

if __name__ == '__main__':
    main()
