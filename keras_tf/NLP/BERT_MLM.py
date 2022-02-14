""" pre-trained BERT with Masked LM task
    paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    see: https://arxiv.org/pdf/1810.04805.pdf

    dataset: Large Movie Review Dataset v1.0
    download: https://ai.stanford.edu/~amaas/data/sentiment/
    paper: Learning Word Vectors for Sentiment Analysis
    see: https://aclanthology.org/P11-1015.pdf
    For more information about the dataset please see the README file
"""

import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Add, MultiHeadAttention, LayerNormalization, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, regexp_tokenize
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
        def tokenize(line):
            # some regex may be useful:
            # '(?:[A-Za-z]\.)'  # abbreviations(both upper and lower case, like "e.g.", "U.S.A.")
            # '\w+(?:-\w+)*'        # words with optional internal hyphens
            line = '[cls] ' + line
            tokens = regexp_tokenize(line, pattern=r'\[[a-zA-Z]+\]|\w+')
            return tokens

        texts = [tokenize(line) for line in self.texts]

        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='[unk]')
        tokenizer.fit_on_texts(texts)

        # --------------------
        # add [mask] tokens to the vocabulary
        vocab = tokenizer.word_index
        vocab[''] = 0

        vocab = sorted(vocab.items(), key=lambda kv: kv[1])

        if max_vocab_size < len(vocab):  # only preserve the max_vocab_size words with highest frequency
            vocab = vocab[:max_vocab_size]

        vocab_size = len(vocab)  # replace the last word-id pair with [mask]-id pair
        vocab[vocab_size - 1] = ('[mask]', vocab_size - 1)

        tokenizer.word_index = dict(vocab)
        # --------------------

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


class TransformerEncoderSublayer(Layer):
    """ sublayer of transformer's encoder
        using hyperparameters of the base model described in paper as default
    """

    def __init__(self, latent_dim, num_heads=4, key_dim=32):
        """
        :param latent_dim: input and output dimensions
        :param num_heads: number of heads for multi-head attention in each sublayer
        :param key_dim: the dimensions in multi-head attention after projection step
        """
        super().__init__()

        self.multihead_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.feedforward = Sequential([
            Dense(2 * latent_dim, activation='relu'),
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


class CosinePositionalEmbedding(Layer):
    """ positional embedding using cosine functions
    """
    def __init__(self, mxlen, latent_dim):
        super().__init__()

        encoding_matrix = np.array([
            [pos / np.power(10000, 2 * (j // 2) / latent_dim) for j in range(latent_dim)]
            for pos in range(mxlen)
        ])
        encoding_matrix[:, 0::2] = np.sin(encoding_matrix[:, 0::2])  # dim 2i
        encoding_matrix[:, 1::2] = np.cos(encoding_matrix[:, 1::2])  # dim 2i+1

        self.embedding_pos = Embedding(
            mxlen, latent_dim, embeddings_initializer=keras.initializers.constant(encoding_matrix))

    def call(self, inputs):
        seq_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=seq_length, delta=1)

        return self.embedding_pos(positions)

class BaseBERT(Model):
    def __init__(self, latent_dim, num_sublayer, vocab_size, mxlen):
        super().__init__()

        self.embedding_token = Embedding(vocab_size, latent_dim)
        self.embedding_pos = CosinePositionalEmbedding(mxlen, latent_dim)

        self.encoder_sublayer = [TransformerEncoderSublayer(latent_dim) for _ in range(num_sublayer)]

    def call(self, inputs):
        embedded_token = self.embedding_token(inputs)
        embedded_pos = self.embedding_pos(inputs)
        x = embedded_token + embedded_pos

        for sublayer in self.encoder_sublayer:
            x = sublayer(x)

        return x

class PretrainBERT:
    def __init__(self, latent_dim=128, num_sublayer=2):
        imdb = ACLIMDbPreprocessor(r'../input/aclimdb/aclImdb', num_samples=1000)

        self.texts = imdb.texts
        self.word_seq = imdb.word_seq
        self.vocab, self.vocab_rev = imdb.vocab, imdb.vocab_rev
        self.vocab_size = len(self.vocab.keys())

        self.inputs, self.sample_weights = self.getPretrainData(self.word_seq)
        self.outputs = self.word_seq.copy()

        self.baseModel = BaseBERT(
            latent_dim, num_sublayer, self.vocab_size, self.inputs.shape[1])

        self.pretrainModel = self.getPretrainModel(self.vocab_size)

    def getPretrainData(self, word_seq):
        # randomly mask 15% of tokens
        mask = np.random.rand(*word_seq.shape) < 0.15

        # do not mask special tokens
        mask[word_seq <= 1] = False
        mask[word_seq == self.vocab['[cls]']] = False

        # BERT only predicts the masked words rather than reconstructing the entire input
        # we can use the "sample_weights" argument of the "fit" method to achieve that
        sample_weights = np.zeros(word_seq.shape)
        sample_weights[mask] = 1

        # construct masked input
        # For the chosen position to be masked, replace 90% of tokens with [mask] token
        # the left 10% of tokens are remained unchanged
        # then replace 1/9 of [mask] tokens with random token
        seq_mask = word_seq.copy()
        mask_pos = mask & (np.random.rand(*word_seq.shape) < 0.9)
        seq_mask[mask_pos] = self.vocab['[mask]']

        random_pos = mask_pos & (np.random.rand(*word_seq.shape) < 1 / 9)
        seq_mask[random_pos] = np.random.randint(0, self.vocab_size - 1, np.sum(random_pos))

        return seq_mask, sample_weights

    def getPretrainModel(self, vocab_size):
        inputs = Input(shape=(None, ))

        base_outputs = self.baseModel(inputs)
        outputs = Dense(vocab_size, activation='softmax')(base_outputs)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', weighted_metrics=['acc'])
        return model

    def preTrain(self, epochs, batch_size):
        self.pretrainModel.fit(self.inputs, self.outputs, sample_weight=self.sample_weights,
                       epochs=epochs, batch_size=batch_size)

        self.baseModel.save('./BaseBERT')

        self.pretrainValidate()

    def pretrainValidate(self):
        num = 5

        random_pos = np.random.randint(0, self.inputs.shape[0], num)
        inputs = self.inputs[random_pos, :]

        outputs = self.pretrainModel.predict(inputs)
        outputs = np.argmax(outputs, axis=2)

        res = inputs.copy()
        mask_pos = np.where(self.sample_weights[random_pos, :] == 1)
        res[mask_pos] = outputs[mask_pos]

        for i in range(num):
            sentence = self.texts[random_pos[i]]
            masked_sentence = ''.join([self.vocab_rev[idx] + ' ' for idx in inputs[i]])
            decoded_sentence = ''.join([self.vocab_rev[idx] + ' ' for idx in res[i]])

            print('-----------')
            print('sentence:', sentence)
            print('masked sentence:', masked_sentence)
            print('decoded sentence:', decoded_sentence)


pretrain_bert = PretrainBERT()
pretrain_bert.preTrain(epochs=20, batch_size=32)
