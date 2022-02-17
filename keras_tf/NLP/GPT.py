""" GPT (Generative Pre-Training) model
    paper: Improving Language Understanding by Generative Pre-Training
    see: https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf

    dataset: Large Movie Review Dataset v1.0
    download: https://ai.stanford.edu/~amaas/data/sentiment/
    paper: Learning Word Vectors for Sentiment Analysis
    see: https://aclanthology.org/P11-1015.pdf
    For more information about the dataset please see the README file
"""

import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, Embedding, Add, MultiHeadAttention, LayerNormalization, Dropout, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import glob


class ACLIMDbPreprocessor:
    """ preprocesser for the Large Movie Review Dataset v1.0
    """
    def __init__(self, dataDir, num_samples=None, max_vocab_size=30000):
        self.texts = self.getTextList(dataDir, num_samples)

        self.word_seq, self.vocab = self.buildVocabulary(max_vocab_size)

        self.vocab_rev = dict((id, word) for word, id in self.vocab.items())

    def buildVocabulary(self, max_vocab_size):
        texts = [word_tokenize(line) for line in self.texts]

        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='[unk]')
        tokenizer.fit_on_texts(texts)

        tokenizer.word_index[''] = 0

        word_seq = tokenizer.texts_to_sequences(texts)
        word_seq = pad_sequences(word_seq, padding='post', truncating='post')

        return word_seq, tokenizer.word_index

    def getTextList(self, dataDir, num_samples):
        flist = []
        flist += glob.glob(dataDir + "/train/pos/*.txt")
        flist += glob.glob(dataDir + "/train/neg/*.txt")

        flist += glob.glob(dataDir + "/test/pos/*.txt")
        flist += glob.glob(dataDir + "/test/neg/*.txt")

        flist = flist[:min(num_samples, len(flist))]

        texts = []
        for fpath in flist:
            with open(fpath, encoding='UTF-8') as f:
                for line in f:
                    line = line.replace('<br />', ' ')
                    texts += sent_tokenize(line)  # split paragraph to sentence

        return texts


class TransformerBlock(Layer):
    """ sublayer of transformer's decoder
        using hyperparameters of the base model described in paper as default
    """

    def __init__(self, latent_dim, num_heads=4, key_dim=64, ff_dim=512, drop_rate=0.1):
        """
        :param latent_dim: input and output dimensions
        :param num_heads: number of heads for multi-head attention in each sublayer
        :param key_dim: the dimensions in multi-head attention after projection step
        :param ff_dim: units number of feedforward network
        """
        super().__init__()

        self.multihead_atten_mask = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.feedforward = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(latent_dim)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)

    def call(self, inputs):
        future_mask = self.getFeatureMask(inputs)

        x = self.multihead_atten_mask(inputs, inputs, attention_mask=future_mask)
        x = self.dropout1(x)
        x = Add()([x, inputs])
        outputs_atten = self.layernorm1(x)

        x = self.feedforward(outputs_atten)
        x = self.dropout2(x)
        x = Add()([x, outputs_atten])
        outputs = self.layernorm2(x)

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


class TokenAndPositionEmbedding(Layer):
    """ learned positional embedding
    """
    def __init__(self, vocab_size, mxlen, latent_dim):
        super().__init__()
        self.embedding_token = Embedding(vocab_size, latent_dim)
        self.embedding_pos = Embedding(mxlen, latent_dim)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=seq_length, delta=1)

        embedded_token = self.embedding_token(inputs)
        embedded_pos = self.embedding_pos(positions)

        return embedded_token + embedded_pos


class BaseGPT(Model):
    def __init__(self, latent_dim, num_sublayer, vocab_size, mxlen):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(vocab_size, mxlen, latent_dim)
        self.encoder_sublayer = [TransformerBlock(latent_dim) for _ in range(num_sublayer)]

    def call(self, inputs):
        x = self.embedding(inputs)
        for sublayer in self.encoder_sublayer:
            x = sublayer(x)

        return x


class GPT:
    def __init__(self):
        imdb = ACLIMDbPreprocessor(r'../input/aclimdb/aclImdb', num_samples=1000)

        self.texts = imdb.texts
        self.word_seq = imdb.word_seq
        self.vocab, self.vocab_rev = imdb.vocab, imdb.vocab_rev
        self.vocab_size = len(self.vocab.keys())

        self.baseModel = BaseGPT(
            latent_dim=256, num_sublayer=2,
            vocab_size=self.vocab_size, mxlen=self.word_seq.shape[1]
        )

        self.pretrainModel = self.getPretrainModel()

    def getPretrainModel(self):
        inputs = Input(shape=(None, ))

        x = self.baseModel(inputs)
        x = Dense(self.vocab_size)(x)

        model = Model(inputs, x)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss, metrics=['acc'])

        return model

    def preTrain(self, epochs, batch_size):
        inputs = self.word_seq[:, :-1]
        outputs = self.word_seq[:, 1:]

        self.pretrainModel.fit(inputs, outputs, epochs=epochs, batch_size=batch_size)
        self.baseModel.save('./BaseGPT')

        self.pretrainValidate()

    def pretrainValidate(self):
        num = 5

        random_pos = np.random.randint(0, self.word_seq.shape[0], num)
        inputs = self.word_seq[random_pos, :]

        outputs = self.pretrainModel.predict(inputs)
        outputs = np.argmax(outputs, axis=2)

        for i in range(num):
            sentence = self.texts[random_pos[i]]
            decoded_sentence = ''.join([self.vocab_rev[idx] + ' ' for idx in outputs[i]])

            print('-----------')
            print('sentence:', sentence)
            print('decoded sentence:', decoded_sentence)


gpt = GPT()
gpt.preTrain(epochs=10, batch_size=32)
