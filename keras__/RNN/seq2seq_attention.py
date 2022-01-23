""" Seq2Seq with Attention
    paper: Neural Machine Translation By Jointly Learning To Align And Translate
    see: https://arxiv.org/pdf/1409.0473.pdf
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, \
    TimeDistributed, Attention, Average
import numpy as np
from keras__.RNN.PreProcessor import PreProcessor


class Seq2Seq:
    def __init__(self):
        preprocessor = PreProcessor(dataDir='D:\\wallpaper\\datas\\fra-eng\\fra.txt', num_samples=10000)

        self.text_en, self.text_fra = preprocessor.getOriginalText()
        (self.dict_en, self.dict_en_rev), (self.dict_fra, self.dict_fra_rev) = preprocessor.getVocab()
        num_word_en, num_word_fra = preprocessor.getNumberOfWord()
        self.tensor_input, self.tensor_output = preprocessor.getPaddedSeq()

        self.encoder, self.decoder, self.model = self.buildNet(num_word_en, num_word_fra, 256)

    def buildEncoder(self, num_word, latent_dim):
        inputs = Input(shape=(None,))  # shape: (samples, max_length)
        embedded = Embedding(num_word, 128)(inputs)  # shape: (samples, length, vec_dim)

        outputs, h1, c1, h2, c2 = Bidirectional(
            LSTM(latent_dim, return_sequences=True, return_state=True),
            merge_mode='ave'
        )(embedded)

        state_h = Average()([h1, h2])
        state_c = Average()([c1, c2])

        # only save the last state of encoder
        return Model(inputs, [outputs, state_h, state_c])

    def buildDecoder(self, num_word, latent_dim):
        inputs = Input(shape=(None,))   # shape: (samples, max_length)
        embedded = Embedding(num_word, 128)(inputs)  # shape: (samples, length, vec_dim)

        input_state_h = Input(shape=(latent_dim,))
        input_state_c = Input(shape=(latent_dim,))
        lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

        # initial_state(Call arguments): List of initial state tensors to be passed to the first call of the cell
        # Here we use the last state of encoder as the initial state of decoder
        outputs_dec, output_state_h, output_state_c = lstm(
            embedded, initial_state=[input_state_h, input_state_c]
        )

        outputs_enc = Input(shape=(None, latent_dim))
        outputs_atten = Attention()([outputs_dec, outputs_enc])

        prob = TimeDistributed(Dense(num_word, activation='softmax'))(outputs_atten)

        return Model(
            [inputs, input_state_h, input_state_c, outputs_enc],
            [prob, output_state_h, output_state_c]
        )

    def buildNet(self, num_word_in, num_word_out, latent_dim):
        encoder = self.buildEncoder(num_word_in, latent_dim)
        decoder = self.buildDecoder(num_word_out, latent_dim)

        inputs_encoder = Input(shape=(None,))
        inputs_decoder = Input(shape=(None,))

        outputs_encoder, state_h, state_c = encoder(inputs_encoder)
        prob, _, _ = decoder([inputs_decoder, state_h, state_c, outputs_encoder])

        model = Model([inputs_encoder, inputs_decoder], prob)

        # there's no need to pass one-hot tensor when using sparse_categorical_crossentropy
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return encoder, decoder, model

    def trainModel(self, epochs, batch_size):
        # there's one timestep shift when using teach forcing
        outputs_shift = np.zeros(self.tensor_output.shape)
        outputs_shift[:, :-1] = self.tensor_output.copy()[:, 1:]

        self.model.fit(
            [self.tensor_input, self.tensor_output], outputs_shift,
            epochs=epochs, batch_size=batch_size, validation_split=0.2,
        )

        self.test()

    def test(self):
        for idx in range(5):
            input_seq = self.tensor_input[idx: idx + 1]
            translated = self.translate(input_seq)
            print('-')
            print('Input sentence:', self.text_en[idx])
            print('Decoded sentence:', translated)
            print('Ground truth:', self.text_fra[idx])

    def translate(self, input_seq):
        outputs_encoder, state_h, state_c = self.encoder.predict(input_seq)

        # blank target sentence, which only has a <sos> symbol
        cur_word = np.zeros((1, 1))
        cur_word[0, 0] = self.dict_fra['\t']

        max_length = 80
        translated = ''
        for _ in range(max_length):
            outputs, state_h, state_c = self.decoder.predict([cur_word, state_h, state_c, outputs_encoder])

            output_idx = np.argmax(outputs[0, -1, :])
            output_word = self.dict_fra_rev[output_idx]

            # stop when <eos> symbol has been generated
            if output_word == '\n':
                break

            translated += ' ' + output_word

            # next input of decoder
            cur_word = np.zeros((1, 1))
            cur_word[0, 0] = output_idx

        return translated

seq2seq = Seq2Seq()
seq2seq.trainModel(epochs=10, batch_size=64)