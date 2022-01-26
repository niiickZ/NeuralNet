""" getting started to keras application: VGG16
    paper: Very Deep Convolutional Networks For Large-Scale Image Recognition
    see: https://arxiv.org/pdf/1409.1556.pdf(2014.pdf
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

def BuildNet():
    # VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    VGGmodel = VGG16(
        weights='imagenet',  # pre-training on ImageNet
        include_top=False,  # whether containing linear layer on top or not
        input_shape=(224, 224, 3)
        # class=1000 # only to be specified if `include_top` is False
    )

    TopModel = Sequential()
    TopModel.add(Flatten(input_shape=VGGmodel.output_shape[1:]))

    TopModel.add(Dense(4096, activation='relu'))
    TopModel.add(Dropout(0.5))

    TopModel.add(Dense(4096, activation='relu'))
    TopModel.add(Dropout(0.5))

    TopModel.add(Dense(1000, activation='relu'))
    TopModel.add(Dropout(0.5))

    TopModel.add(Dense(1000, activation='softmax'))

    model = Sequential()
    model.add(VGGmodel)
    model.add(TopModel)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()

    return model

def GenerateData(datadir):
    ImageGen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
    )

    DataTrain = ImageGen.flow_from_directory(
        directory=datadir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        subset='training'
    )
    DataValidate = ImageGen.flow_from_directory(
        directory=datadir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=64,
        subset='validation'
    )

    return DataTrain, DataValidate