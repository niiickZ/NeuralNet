""" Auxiliary Classifier GANs
    paper: Conditional Image Synthesis with Auxiliary Classifier GANs
    see: http://proceedings.mlr.press/v70/odena17a/odena17a.pdf
"""

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input, \
    LeakyReLU, Embedding, multiply, Activation, Conv2DTranspose, Conv2D, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class ACGAN:
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

        model.add(Dense(input_dim=self.latent_dim, units=7 * 7 * 256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Reshape((7, 7, 256)))

        model.add(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'))
        model.add(Activation('tanh'))

        noise = Input(shape=(self.latent_dim,))

        label = Input(shape=(1,))
        labelEmbeded = Flatten()(Embedding(self.num_class, self.latent_dim)(label))
        # Embedding: (samples, length) -> (samples, length, latent_dim)

        input = multiply([noise, labelEmbeded])
        img = model(input)

        return Model([noise, label], img)

    def buildDiscriminator(self):
        model = Sequential()

        model.add(Conv2D(input_shape=self.img_shape, filters=64, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Flatten())

        img = Input(shape=self.img_shape)
        x = model(img)

        label_pred = Dense(10, activation='softmax')(x)  # auxiliary classifier
        validity = Dense(1, activation='sigmoid')(x)

        discriminator = Model(img, [validity, label_pred])
        discriminator.compile(
            optimizer=Adam(2e-4),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )  # there's no need to pass one-hot tensor when using sparse_categorical_crossentropy
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

        validity, label_pred = self.discriminator(img)

        self.combined = Model([noise, label], [validity, label_pred])
        self.combined.compile(
            optimizer=Adam(2e-4),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

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
            randomLabel = np.random.randint(0, 10, (batch_size, 1))

            # generate images with noise and label
            genImgs = self.generator.predict([noise, randomLabel])

            # train the discriminator
            D_loss_real = self.discriminator.train_on_batch(orgImgs, [valid, label])
            D_loss_fake = self.discriminator.train_on_batch(genImgs, [fake, randomLabel])
            D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

            # train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            randomLabel = np.random.randint(0, 10, (batch_size, 1))
            G_loss = self.combined.train_on_batch([noise, randomLabel], [valid, randomLabel])

            print("epoch {} --- D_loss: {:.4f}, classify loss: {:.4f}, G loss: {:.4f}".format(
                epoch, D_loss[1], D_loss[2], np.mean(G_loss)))

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
    acgan = ACGAN()
    acgan.trainModel(epochs=10000)

if __name__ == '__main__':
    main()
