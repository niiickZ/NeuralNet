""" speech transformer
    paper: Speech-Transformer——A No-Recurrence Sequence-to-Sequence Model for Speech Recognition
    see: https://ieeexplore.ieee.org/abstract/document/8462506
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Layer, Attention, Reshape, Add, \
    Conv2D, TimeDistributed, MultiHeadAttention, Dense, Embedding, LayerNormalization
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import numpy as np
from keras_tf.ASR.preprocessor import LJSpeechPreprocessor
import math


class Dataloader(Sequence):
    """dataloader for speech transformer"""
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


class MultiHeadAttention2D(Layer):
    def __init__(self, num_heads):
        """A simple implementation of 2D Multi-Head Attention proposed in paper
        "Speech-Transformer——A No-Recurrence Sequence-to-Sequence Model for Speech Recognition"

        Note that the MultiHeadAttention2D can only be used as self-attention,
        since it contains a transpose operation in "frequency attention"

        Argument:
        :param num_heads: number of attention heads, i.e. number of filters of convolution
        """
        super().__init__()

        self.num_heads = num_heads
        self.conv_Q = Conv2D(num_heads, 5, padding='same')
        self.conv_V = Conv2D(num_heads, 5, padding='same')
        self.conv_K = Conv2D(num_heads, 5, padding='same')
        self.conv_out = Conv2D(1, 5, padding='same')

    def call(self, query, value, key=None):
        """
        :param query: Query Tensor of shape (batch_size, Tq, dim)
        :param value: Value Tensor of shape (batch_size, Tv, dim)
        :param key: Key Tensor of shape (batch_size, Tv, dim). If not
        given, will use 'value' for both 'key' and 'value'
        """
        if not key:
            key = value

        # expand (channel) dimension to apply convolution
        # shape (batch_size, T, dim) -> (batch_size, T, dim, 1)
        query = K.expand_dims(query, axis=-1)
        value = K.expand_dims(value, axis=-1)
        key = K.expand_dims(key, axis=-1)

        # shape (batch_size, T, dim, num_heads)
        feat_Q = self.conv_Q(query)
        feat_V = self.conv_V(value)
        feat_K = self.conv_K(key)

        # Separate feature maps by channel
        # Then generate a list of tuples of length num_heads
        # like [(Q1, V1, K1), (Q2, V2, K2), ..., (Qn, Vn, Kn)]
        combined = [
            (feat_Q[:, :, :, i], feat_V[:, :, :, i], feat_K[:, :, :, i])
            for i in range(self.num_heads)
        ]

        # transpose feature maps to apply frequency attention
        combined_transpose = [
            (K.permute_dimensions(feat_query, (0, 2, 1)),
             K.permute_dimensions(feat_value, (0, 2, 1)),
             K.permute_dimensions(feat_key, (0, 2, 1)))
            for feat_query, feat_value, feat_key in combined
        ]

        out_temporal_atten = [
            Attention()([feat_query, feat_value, feat_key])
            for feat_query, feat_value, feat_key in combined
        ]

        out_frequncy_atten = [
            Attention()([feat_query_trans, feat_value_trans, feat_key_trans])
            for feat_query_trans, feat_value_trans, feat_key_trans in combined_transpose
        ]

        # concatenate feature maps by channel
        feat_time = K.concatenate([K.expand_dims(feat, -1) for feat in out_temporal_atten], axis=-1)
        feat_freq = K.concatenate([K.expand_dims(feat, -1) for feat in out_frequncy_atten], axis=-1)

        feat = K.concatenate([feat_time, K.permute_dimensions(feat_freq, (0, 2, 1, 3))], axis=-1)
        outputs = self.conv_out(feat)
        outputs = K.squeeze(outputs, axis=-1)

        return outputs


class TransformerEncoderSublayer(Layer):
    """ sublayer of speech transformer's encoder
    """
    def __init__(self, latent_dim, num_heads=16, hidden_dim=1024):
        """
        :param latent_dim: input and output dimensions of each timestep
        :param num_heads: number of attention heads, i.e. number of filters of
        each convolution network in 2D multi-head attention
        :param hidden_dim: number of units of hidden layer in feedforward network
        """
        super().__init__()

        self.multihead_atten2D = MultiHeadAttention2D(num_heads=num_heads)
        self.feedforward = Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(latent_dim)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs):
        x = self.multihead_atten2D(inputs, inputs)
        x = Add()([x, inputs])
        outputs_atten = self.layernorm1(x)

        x = self.feedforward(outputs_atten)
        x = Add()([x, outputs_atten])
        outputs = self.layernorm2(x)

        return outputs


class TransformerDecoderSublayer(Layer):
    """ sublayer of speech transformer's decoder
    """
    def __init__(self, latent_dim, num_heads=4, key_dim=64, hidden_dim=1024):
        """
        :param latent_dim: input and output dimensions of each timestep
        :param num_heads: number of attention heads
        :param key_dim: output dimensions of projection step in multi-head attention
        :param hidden_dim: number of units of hidden layer in feedforward network
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
    """
    def __init__(self, mxlen, latent_dim):
        super().__init__()
        self.embedding_pos = Embedding(mxlen, latent_dim)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        embedded_pos = self.embedding_pos(positions)

        return embedded_pos


class SpeechTransformer:
    def __init__(self):
        preprocessor = LJSpeechPreprocessor(
            '../input/ljspeech/LJSpeech-1.1', num_samples=None)

        self.wavs_list = preprocessor.getWavsList()
        self.orginal_text = preprocessor.getOriginalText()
        self.target_seq, self.vocab, self.vocab_rev = preprocessor.getTargetSequence(SOS='\t', EOS='\n')
        self.vocab_size = len(self.vocab.keys())

        self.mxlen_audio = 512
        self.mxlen_text = self.target_seq.shape[1]
        self.latent_dim = 128

        self.model = self.buildNet()

    def buildEncoder(self, num_sublayer):
        inputs_audio = Input(shape=(None, self.latent_dim))

        x = K.expand_dims(inputs_audio, -1)
        x = Conv2D(32, kernel_size=[7, 7], strides=2, padding='same', activation='relu')(x)
        x = Conv2D(32, kernel_size=[7, 7], strides=2, padding='same', activation='relu')(x)

        x = Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        x = TimeDistributed(Dense(self.latent_dim, activation='relu'))(x)
        x = x + PositionEmbedding(self.mxlen_audio, self.latent_dim)(x)

        for _ in range(num_sublayer):
            x = TransformerEncoderSublayer(self.latent_dim, hidden_dim=512)(x)

        return Model(inputs_audio, x)

    def buildDecoder(self, num_sublayer):
        inputs_tar = Input(shape=(None,))
        outputs_enc = Input(shape=(None, self.latent_dim))

        embedded_token = Embedding(self.vocab_size, self.latent_dim)(inputs_tar)
        embedded_pos = PositionEmbedding(self.mxlen_text, self.latent_dim)(inputs_tar)
        x = embedded_token + embedded_pos

        for _ in range(num_sublayer):
            x = TransformerDecoderSublayer(self.latent_dim, hidden_dim=512)(x, outputs_enc)

        return Model([inputs_tar, outputs_enc], x)

    def buildNet(self):
        encoder = self.buildEncoder(num_sublayer=3)
        decoder = self.buildDecoder(num_sublayer=3)

        inputs_audio = Input(shape=(None, self.latent_dim))
        inputs_target = Input(shape=(None,))

        outputs_enc = encoder(inputs_audio)
        outputs_dec = decoder([inputs_target, outputs_enc])

        prob = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(outputs_dec)

        model = Model([inputs_audio, inputs_target], prob)

        # there's no need to pass one-hot tensor when using sparse_categorical_crossentropy
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def trainModel(self, epochs, batch_size):
        dataloader = Dataloader(
            self.wavs_list, self.target_seq, self.latent_dim, batch_size=batch_size)

        self.model.fit(dataloader, epochs=epochs)

        self.test()
        self.model.save_weights('./SpeechTransformer.h5')

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
        # blank target sentence, which only has a <sos> symbol
        output_seq = np.zeros((1, 1))
        output_seq[0, 0] = self.vocab['\t']

        max_length = 80
        res = ''
        for _ in range(max_length):
            outputs = self.model.predict([spect, output_seq])

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


model = SpeechTransformer()
model.trainModel(epochs=10, batch_size=64)
# It's suggested to train for around 40 epochs or more for LJSpeech dataset
