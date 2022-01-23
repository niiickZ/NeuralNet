from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


class PreProcessor:
    """
        load english to franch translation dataset
        see: http://www.manythings.org/anki/fra-eng.zip
        find datas more languages: http://www.manythings.org/anki/
    """

    def __init__(self, dataDir, num_samples=None, shuffle=True):
        self.text_en, self.text_fra = self.readData(dataDir, num_samples, shuffle)  # 读取数据

        # 分词并获取索引化文本和word->id字典
        self.seq_en, self.dict_en = self.buildVocab(self.text_en)
        self.seq_fra, self.dict_fra = self.buildVocab(self.text_fra)

        self.dict_en['<unknown>'] = 0
        self.dict_fra['<unknown>'] = 0

    def getOriginalText(self):
        """ 获取原文本数据 """
        return self.text_en, self.text_fra

    def getVocab(self):
        """ 获取 id->word字典 和 word->id字典 """
        dict_en_rev = dict((id, char) for char, id in self.dict_en.items())
        dict_fra_rev = dict((id, char) for char, id in self.dict_fra.items())

        return (self.dict_en, dict_en_rev), (self.dict_fra, dict_fra_rev)

    def getNumberOfWord(self):
        """ 获取分词后不同单词数 """
        num_word_en = len(self.dict_en.keys())
        num_word_fra = len(self.dict_fra.keys())
        return num_word_en, num_word_fra

    def getPaddedSeq(self):
        """ 填充索引化后的句子至相同长度 """
        padded_seq_en = pad_sequences(self.seq_en, padding='post')
        padded_seq_fra = pad_sequences(self.seq_fra, padding='post')
        return padded_seq_en, padded_seq_fra

    def buildVocab(self, text):
        """ 分词并建立词汇表 """
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')  # 需要过滤除\t\n外的特殊字符
        tokenizer.fit_on_texts(text)
        seq = tokenizer.texts_to_sequences(text)
        return seq, tokenizer.word_index

    def readData(self, dataDir, num_samples, shuffle):
        """ 读取数据 """
        with open(dataDir, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if num_samples:
            lines = lines[:min(num_samples, len(lines))]

        if shuffle:
            random.shuffle(lines)

        text_en = [line.split('\t')[0] for line in lines]
        text_fra = ['\t ' + line.split('\t')[1] + ' \n' for line in lines]  # \t和\n分表用作<sos>和<eos>标记

        return text_en, text_fra