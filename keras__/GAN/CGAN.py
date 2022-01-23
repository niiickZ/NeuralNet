""" Conditional GAN
    paper: Conditional Generative Adversarial Nets
    see: https://arxiv.org/pdf/1411.1784.pdf
"""

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, \
    LeakyReLU, Embedding, multiply, Activation, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class CGAN:
    def __init__(self):
        self.img_row = 28
        self.img_col = 28
        self.channel = 1
        self.img_shape = (self.img_row, self.img_col, self.channel)
        self.latent_dim = 100
        self.num_class = 10

        self.buildGAN()

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

        label = Input(shape=(1,))
        labelEmbeded = Flatten()(Embedding(self.num_class, self.latent_dim)(label))
        # Embedding: (samples, length) -> (samples, length, latent_dim)

        input = multiply([noise, labelEmbeded])
        img = model(input)

        return Model([noise, label], img)

    def buildDiscriminator(self):
        model = Sequential()

        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Dense(64))
        model.add(LeakyReLU(0.2))

        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        imgFlat = Flatten()(img)

        label = Input(shape=(1, ))
        labelEmbeded = Flatten()(Embedding(self.num_class, np.prod(self.img_shape))(label))
        # Embedding: (samples, length) -> (samples, length, latent_dim)

        input = multiply([imgFlat, labelEmbeded])
        validity = model(input)

        discriminator = Model([img, label], validity)
        discriminator.compile(optimizer=Adam(2e-4), loss='binary_crossentropy')
        return discriminator

    def buildGAN(self):
        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # setting trainable to false after compiling
        # the discriminator will be frozen only when training the combined
        self.discriminator.trainable = False

        validity = self.discriminator([img, label])

        self.combined = Model([noise, label], validity)
        self.combined.compile(optimizer=Adam(2e-4), loss='binary_crossentropy')

    def trainModel(self, epochs, batch_size=64):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        # Normalize to [-1, 1]
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            epoch += 1

            # randomly select a batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            orgImgs, label = X_train[idx], Y_train[idx].reshape(-1, 1)

            # noise with standard normal distribution and randomly generated category label
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            randomLabel = np.random.randint(0, 10, (batch_size, ))

            # generate images with noise and label
            genImgs = self.generator.predict([noise, randomLabel])

            # train the discriminator
            D_loss_real = self.discriminator.train_on_batch([orgImgs, label], valid)
            D_loss_fake = self.discriminator.train_on_batch([genImgs, randomLabel], fake)
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

            # train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            randomLabel = np.random.randint(0, 10, (batch_size, ))
            G_loss = self.combined.train_on_batch([noise, randomLabel], valid)

            print("epoch {} --- D loss: {:.4f} , G loss: {:.4f}".format(epoch, D_loss, G_loss))

            if epoch % 400 == 0:
                self.saveImage(epoch)


    def saveImage(self, epoch):
        r, c = 3, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        label = np.tile(np.arange(0, 10), 3).reshape(-1, 1)

        genImgs = self.generator.predict([noise, label])

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
    cgan = CGAN()
    cgan.trainModel(epochs=10000)

if __name__ == '__main__':
    main()
