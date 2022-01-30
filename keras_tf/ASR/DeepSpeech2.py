""" Deep Speech2
    paper: Deep Speech 2 - End-to-End Speech Recognition in English and Mandarin
    see: http://proceedings.mlr.press/v48/amodei16.pdf

    see also: Deep Speech - Scaling up end-to-end speech recognition
    https://arxiv.org/pdf/1412.5567.pdf
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Bidirectional, GRU, BatchNormalization, Conv2D, Activation, Reshape
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import numpy as np
import math
from keras_tf.ASR.preprocessor import LJSpeechPreprocessor


class Dataloader(Sequence):
    """dataloader for CTC model"""
    def __init__(self, wavs_list, target_sequence, n_mels, batch_size=64):
        self.wavs_list = wavs_list
        self.targets = target_sequence

        self.n_mels = n_mels
        self.batch_size = batch_size

        self.fnum = len(wavs_list)

    def __len__(self):
        return math.ceil(self.fnum / self.batch_size)

    def __getitem__(self, idx):
        st = idx * self.batch_size
        ed = min((idx + 1) * self.batch_size, self.fnum)

        targets = self.targets[st:ed, :]  # shape (samples, length)
        inputs = LJSpeechPreprocessor.getSpectrograms(
            self.wavs_list[st:ed], self.n_mels
        )  # shape (samples, mxlen, n_mels)

        return inputs, targets


class DeepSpeech2:
    def __init__(self):
        preprocessor = LJSpeechPreprocessor('D:\\wallpaper\\datas\\LJSpeech-1.1', num_samples=None)

        self.wavs_list = preprocessor.getWavsList()
        self.orginal_text = preprocessor.getOriginalText()
        self.target_seq, self.vocab, self.vocab_rev = preprocessor.getTargetSequence()
        self.vocab_size = len(self.vocab.keys())

        self.latent_dim = 128

        self.model = self.buildNet()

    def buildNet(self):
        inputs = Input(shape=(None, self.latent_dim))
        x = Reshape((-1, self.latent_dim, 1))(inputs)

        x = Conv2D(32, kernel_size=[7, 11], strides=[1, 1], padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, kernel_size=[7, 11], strides=[1, 2], padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

        x = Bidirectional(GRU(512, return_sequences=True), merge_mode='sum')(x)
        x = Bidirectional(GRU(512, return_sequences=True), merge_mode='sum')(x)
        x = Bidirectional(GRU(512, return_sequences=True), merge_mode='sum')(x)
        x = BatchNormalization()(x)

        x = Dense(256, activation='relu')(x)
        prob = Dense(self.vocab_size + 1, activation='softmax')(x)

        model = Model(inputs, prob)
        model.compile(optimizer='adam', loss=self.CTCLoss)
        return model

    def CTCLoss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        pred_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        pred_length = pred_length * tf.ones(shape=(batch_size, 1), dtype="int32")
        label_length = label_length * tf.ones(shape=(batch_size, 1), dtype="int32")

        loss = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
        return loss

    def trainModel(self, epochs, batch_size=64):
        dataloader = Dataloader(
            self.wavs_list, self.target_seq, self.latent_dim,
            batch_size=batch_size
        )

        self.model.fit(dataloader, epochs=epochs)
        self.test()

    def test(self):
        for i in range(5):
            inputs = LJSpeechPreprocessor.getSpectrograms(
                self.wavs_list[i:i + 1], self.latent_dim
            )

            res = self.recognize(inputs[0:1])
            print('-')
            print('Decoded Sentence:', res)
            print('Ground Truth:', self.orginal_text[i])

    def recognize(self, spect):
        pred = self.model.predict(spect)
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decode = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output = K.get_value(decode)

        res = ''
        for x in output[0]:
            if x == -1 or x == 0:
                continue
            res += self.vocab_rev[x]

        return res


speechRecognizer = DeepSpeech2()
speechRecognizer.trainModel(epochs=20, batch_size=8)