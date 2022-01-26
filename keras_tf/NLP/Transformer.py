""" Transformer
    paper: Attention is all you need
    see: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Add, MultiHeadAttention, LayerNormalization
import numpy as np
from keras_tf.NLP.preprocessor import TatoebaPreprocessor


class TransformerEncoderSublayer(Layer):
    """ sublayer of transformer's encoder
        using hyperparameters of the base model described in paper as default
    """

    def __init__(self, latent_dim, num_heads=8, key_dim=64, hidden_dim=2048):
        """
        :param latent_dim: input and output dimensions
        :param num_heads: number of heads for multi-head attention in each sublayer
        :param key_dim: the dimensions in multi-head attention after projection step
        :param hidden_dim: units number of hidden layer in feedforward network in each sublayer
        """
        super().__init__()

        self.multihead_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.feedforward = Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(latent_dim)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs):
        x = self.multihead_atten(inputs, inputs)
        x = Add()([x, inputs])
        outputs_atten = self.layernorm1(x)

        x = self.feedforward(outputs_atten)
        x = Add()([x, outputs_atten])
        outputs = self.layernorm2(x)

        return outputs

class TransformerDecoderSublayer(Layer):
    """ sublayer of transformer's decoder
        using hyperparameters of the base model described in paper as default
    """

    def __init__(self, latent_dim, num_heads=8, key_dim=64, hidden_dim=2048):
        """
        :param latent_dim: input and output dimensions
        :param num_heads: number of heads for multi-head attention in each sublayer
        :param key_dim: the dimensions in multi-head attention after projection step
        :param hidden_dim: units number of hidden layer in feedforward network in each sublayer
        """
        super().__init__()

        self.multihead_atten_mask = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.multihead_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.feedforward = Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(latent_dim)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()

    def call(self, inputs, outputs_enc):
        future_mask = self.getFeatureMask(inputs)

        x = self.multihead_atten_mask(inputs, inputs, attention_mask=future_mask)
        x = Add()([x, inputs])
        outputs_atten_mask = self.layernorm1(x)

        x = self.multihead_atten(inputs, outputs_enc)
        x = Add()([x, outputs_atten_mask])
        outputs_atten = self.layernorm2(x)

        x = self.feedforward(outputs_atten)
        x = Add()([x, outputs_atten])
        outputs = self.layernorm3(x)

        return outputs

    def getFeatureMask(self, inputs):
        """ future mask for self-attention
            return a lower triangular matrix with shape (samples, T_Q, T_K) where T_Q = T_K
        """
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]

        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)

        # generate lower triangular matrix. shape: (length, length)
        mask = tf.cast(i >= j, dtype="int32")

        # tile to shape (batch_size, length, length)
        mask = tf.reshape(mask, (1, seq_length, seq_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )

        return tf.tile(mask, mult)

class PositionEmbedding(Layer):
    """ positional embedding layer
        Including a token embedding and a positional embedding that output
        same dimensions. The two output tensor is then added to one.
    """
    def __init__(self, num_word, mxlen, latent_dim):
        super().__init__()

        self.embedding_token = Embedding(num_word, latent_dim)
        self.embedding_pos = Embedding(mxlen, latent_dim)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=seq_length, delta=1)

        embedded_token = self.embedding_token(inputs)
        embedded_pos = self.embedding_pos(positions)

        return embedded_token + embedded_pos

class Transformer:
    def __init__(self):
        preprocessor = TatoebaPreprocessor(dataDir='D:\\wallpaper\\datas\\fra-eng\\fra.txt')

        self.text_en, self.text_fra = preprocessor.getOriginalText()
        (self.dict_en, self.dict_en_rev), (self.dict_fra, self.dict_fra_rev) = preprocessor.getVocab()
        num_word_en, num_word_fra = preprocessor.getNumberOfWord()
        self.tensor_input, self.tensor_output = preprocessor.getPaddedSeq()

        mxlen_en = self.tensor_input.shape[-1]
        mxlen_fra = self.tensor_output.shape[-1]

        self.buildNet(
            num_word_en, num_word_fra,
            mxlen_en, mxlen_fra
        )

    def buildEncoder(self, latent_dim, num_sublayer):
        inputs = Input(shape=(None, latent_dim))

        x = TransformerEncoderSublayer(latent_dim)(inputs)
        for _ in range(num_sublayer - 1):
            x = TransformerEncoderSublayer(latent_dim)(x)

        return Model(inputs, x)

    def buildDecoder(self, latent_dim, num_sublayer):
        inputs = Input(shape=(None, latent_dim))
        outputs_enc = Input(shape=(None, latent_dim))

        x = TransformerDecoderSublayer(latent_dim)(inputs, outputs_enc)
        for _ in range(num_sublayer - 1):
            x = TransformerDecoderSublayer(latent_dim)(inputs, outputs_enc)

        return Model([inputs, outputs_enc], x)

    def buildNet(self, num_word_in, num_word_out, mxlen_in, mxlen_out, latent_dim=512, num_sublayer=6):
        self.encoder = self.buildEncoder(latent_dim, num_sublayer)
        self.decoder = self.buildDecoder(latent_dim, num_sublayer)

        inputs = Input(shape=(None,))
        targets = Input(shape=(None,))

        embedd_inputs = PositionEmbedding(num_word_in, mxlen_in, latent_dim)(inputs)
        embedd_targets = PositionEmbedding(num_word_out, mxlen_out, latent_dim)(targets)

        outputs_enc = self.encoder(embedd_inputs)
        outputs_dec = self.decoder([embedd_targets, outputs_enc])

        prob = Dense(num_word_out, activation='softmax')(outputs_dec)

        self.model = Model([inputs, targets], prob)

        # there's no need to pass one-hot tensor when using sparse_categorical_crossentropy
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def trainModel(self, epochs, batch_size):
        # there's one timestep shift when using teach forcing
        outputs_shift = np.zeros(self.tensor_output.shape)
        outputs_shift[:, :-1] = self.tensor_output.copy()[:, 1:]

        self.model.fit(
            [self.tensor_input, self.tensor_output], outputs_shift,
            epochs=epochs, batch_size=batch_size, validation_split=0.2,
        )

        self.test()

        self.model.save_weights('./transformer.h5')

    def test(self):
        for idx in range(5):
            input_seq = self.tensor_input[idx: idx + 1]
            translated = self.translate(input_seq)
            print('-')
            print('Input sentence:', self.text_en[idx])
            print('Decoded sentence:', translated)
            print('Ground truth:', self.text_fra[idx][1:])

    def translate(self, input_seq):
        # the current word is <sos>
        output_seq = np.zeros((1, 1))
        output_seq[0, 0] = self.dict_fra['\t']

        max_length = 80
        translated = ''
        for _ in range(max_length):
            pred = self.model.predict([input_seq, output_seq])

            token_idx = np.argmax(pred[0, -1, :])
            token = self.dict_fra_rev[token_idx]

            # stop when <eos> has been generated
            if token == '\n':
                break

            translated += ' ' + token

            output_seq = np.hstack((output_seq, np.zeros((1, 1))))
            output_seq[0, -1] = token_idx

        return translated


transformer = Transformer()
transformer.trainModel(epochs=20, batch_size=64)
