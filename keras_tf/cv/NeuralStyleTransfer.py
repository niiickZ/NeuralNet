""" Neural Style Transfer
    paper: A neural algorithm of artistic style
    see: https://arxiv.org/pdf/1508.06576.pdf

    paper: Image Style Transfer Using Convolutional Neural Networks
    see: https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

    Implementation reference: https://keras.io/examples/generative/neural_style_transfer/
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import vgg19
import tensorflow.keras.backend as K
import numpy as np
import cv2

def gram_matrix(x):
    x = K.permute_dimensions(x, (2, 0, 1))
    x = K.reshape(x, (K.shape(x)[0], -1))
    gram = K.dot(x, K.transpose(x))
    return gram

def style_loss(style, combination):
    gram_style = gram_matrix(style)
    gram_combination = gram_matrix(combination)

    h, w, c = style.get_shape().as_list()
    fac = 4.0 * ((h * w) ** 2) * (c ** 2)

    return K.sum(K.square(gram_style - gram_combination)) / fac


def content_loss(content, combination):
    return K.sum(K.square(content - combination)) / 2.0


def calc_loss(outputs):
    content_weight = 2e-4
    style_weight = 1

    content_layer_name = 'block5_conv2'
    style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    loss = K.variable(0.)

    features = outputs[content_layer_name]
    loss = loss + content_weight * content_loss(features[0, :, :, :], features[2, :, :, :])

    for layer_name in style_layer_names:
        features = outputs[layer_name]
        sloss = style_loss(features[1, :, :, :], features[2, :, :, :])

        loss = loss + style_weight / len(style_layer_names) * sloss

    return loss


def preprocessImage(fpath, shape):
    img = cv2.imread(fpath, 1)
    img = cv2.resize(img, shape)

    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = img / 127.5 - 1

    return img

def saveImage(img, fpath):
    img = (img + 1) * 127.5
    img = np.clip(img, 0, 255).astype("uint8")
    cv2.imwrite(fpath, img)

def getFeatureExtractor():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    return Model(vgg.inputs, outputs_dict)

def train(img_content, img_style, img_combination):
    extractor = getFeatureExtractor()
    optimizer = RMSprop()

    epochs = 8000
    for step in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            inputs = K.concatenate([img_content, img_style, img_combination], axis=0)
            outputs = extractor(inputs)
            loss = calc_loss(outputs)

        grad = tape.gradient(loss, img_combination)

        optimizer.apply_gradients([(grad, img_combination)])

        if step % 1000 == 0:
            print("epoch {}: loss={:.4f}".format(step, loss))
            saveImage(img_combination.numpy()[0], './epoch{}.png'.format(step))

fpath_content = '../input/styletransfer/content.jpg'
fpath_style = '../input/styletransfer/style.jpg'

h, w = cv2.imread(fpath_content, 0).shape
shape = (w, h)

img_content = K.variable(preprocessImage(fpath_content, shape))
img_style = K.variable(preprocessImage(fpath_style, shape))
img_combination = K.variable(preprocessImage(fpath_content, shape))

train(img_content, img_style, img_combination)
