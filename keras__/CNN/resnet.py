""" ResNet 34-layer
    paper: Deep Residual Learning for Image Recognition
    see: https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
"""

from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Activation, BatchNormalization, \
    MaxPool2D, GlobalAveragePooling2D, Add
from keras.optimizers import Adam

class ResNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.resnet = self.buildNet()
        self.resnet.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')

        print(self.resnet.summary())

    def buildNet(self):
        inputs = Input(self.input_shape)

        x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
        x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

        num_block = [3, 4, 6, 3]
        for k in range(4):
            for idx in range(num_block[k]):
                downsample = (idx==0 and k!=0)
                x = self.resblock(inputs=x, filters=64*2**k, downsample=downsample)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(1000, activation='softmax')(x)

        return Model(inputs, outputs)

    def resblock(self, inputs, filters, downsample=False):
        strides = 2 if downsample else 1

        x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if downsample:
            inputs = Conv2D(filters, kernel_size=1, strides=2, padding='same', activation='relu')(inputs)

        x = Add()([inputs, x])

        outputs = Activation('relu')(x)
        return outputs

if __name__ == '__main__':
    resnet = ResNet()