""" LAS speech recognition model
    paper: Listen, Attend and Spell
    see: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/44926.pdf
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, LSTM, Bidirectional, Embedding, Attention, TimeDistributed, Dense
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import math
import numpy as np
from keras_tf.ASR.preprocessor import LJSpeechPreprocessor


class Dataloader(Sequence):
    """dataloader for LAS"""
    def __init__(self, wavs_list, target_sequence, n_mels, batch_size=64):
        self.wavs_list = wavs_list
        self.targets = target_sequence

        self.n_mels = n_mels
        self.batch_size = batch_size

        self.fnum = len(wavs_list)

        # there's one timestep shift for the ground truth
        self.targets_shift = np.zeros(self.targets.shape)
        self.targets_shift[:, :-1] = self.targets.copy()[:, 1:]

    def __len__(self):
        return math.ceil(self.fnum / self.batch_size)

    def __getitem__(self, idx):
        st = idx * self.batch_size
        ed = min((idx + 1) * self.batch_size, self.fnum)

        targets = self.targets[st:ed, :]  # shape (samples, length)
        targets_shift = self.targets_shift[st:ed, :]  # shape (samples, length)
        inputs = LJSpeechPreprocessor.getSpectrograms(
            self.wavs_list[st:ed], self.n_mels
        )  # shape (samples, mxlen, n_mels)

        return [inputs, targets], targets_shift


class ConsecutiveConcat(Layer):
    """Concatenate tensors of adjacent timestep.
    Used to create a pyramidal RNN.
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        if K.shape(inputs)[1] % 2 == 1:
            inputs = K.temporal_padding(inputs, (0, 1))

        x = K.concatenate([inputs[:, 0::2, :], inputs[:, 1::2, :]], axis=-1)
        return x


class LAS:
    def __init__(self):
        self.input_dim = 128

        preprocessor = LJSpeechPreprocessor(
            '../input/ljspeech/LJSpeech-1.1', num_samples=None)

        self.wavs_list = preprocessor.getWavsList()
        self.orginal_text = preprocessor.getOriginalText()
        self.target_seq, self.vocab, self.vocab_rev = preprocessor.getTargetSequence(SOS='\t', EOS='\n')
        self.vocab_size = len(self.vocab.keys())

        self.listener, self.speller, self.model = self.buildNet()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    def buildNet(self):
        listener = self.buildListener()
        speller = self.buildSpeller()

        inputs_audio = Input(shape=(None, self.input_dim))
        inputs_target = Input(shape=(None, ))

        outputs_listener = listener(inputs_audio)
        prob = speller([inputs_target, outputs_listener])

        return listener, speller, Model([inputs_audio, inputs_target], prob)

    def buildListener(self):
        inputs = Input(shape=(None, self.input_dim))

        # pyramidal bidirectional LSTM
        x = Bidirectional(LSTM(256, return_sequences=True), merge_mode='ave')(inputs)
        x = ConsecutiveConcat()(x)

        x = Bidirectional(LSTM(256, return_sequences=True), merge_mode='ave')(x)
        x = ConsecutiveConcat()(x)

        x = Bidirectional(LSTM(256, return_sequences=True), merge_mode='ave')(x)
        outputs = ConsecutiveConcat()(x)

        return Model(inputs, outputs)

    def buildSpeller(self):
        inputs = Input(shape=(None, ))
        outputs_listener = Input(shape=(None, 512))

        embedded = Embedding(self.vocab_size, 128)(inputs)
        x = LSTM(256, return_sequences=True)(embedded)
        x = LSTM(512, return_sequences=True)(x)

        x = Attention()([x, outputs_listener])

        prob = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)

        return Model([inputs, outputs_listener], prob)

    def trainModel(self, epochs, batch_size=64):
        dataloader = Dataloader(
            self.wavs_list, self.target_seq, self.input_dim, batch_size=batch_size)

        self.model.fit(dataloader, epochs=epochs)
        self.test()

    def test(self):
        for i in range(5):
            inputs = LJSpeechPreprocessor.getSpectrograms(
                self.wavs_list[i:i + 1], self.input_dim
            )

            res = self.recognize(inputs[0:1])
            print('-')
            print('Decoded Sentence:', res)
            print('Ground Truth:', self.orginal_text[i])

    def recognize(self, spect):
        outputs_listener = self.listener.predict(spect)

        # blank target sentence, which only has a <sos> symbol
        output_seq = np.zeros((1, 1))
        output_seq[0, 0] = self.vocab['\t']

        max_length = 80
        res = ''
        for _ in range(max_length):
            outputs = self.speller.predict([output_seq, outputs_listener])

            output_idx = np.argmax(outputs[0, -1, :])
            output_word = self.vocab_rev[output_idx]

            # stop when <eos> symbol has been generated
            if output_word == '\n':
                break

            res += output_word

            # next input of decoder
            output_seq = np.hstack((output_seq, np.zeros((1, 1))))
            output_seq[0, -1] = output_idx

        return res

las = LAS()
las.trainModel(epochs=25, batch_size=64)