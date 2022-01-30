from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import librosa
import os

class LJSpeechPreprocessor():
    """ load and preprocess the LJSpeech-1.1 dataset
        see: https://keithito.com/LJ-Speech-Dataset/
    """

    def __init__(self, dataDir, num_samples=None):
        self.dataDir = dataDir
        self.metadata = self.readMetadata(num_samples)

    def readMetadata(self, num_samples):
        """Read meta data"""
        fpath = os.path.join(self.dataDir, 'metadata.csv')
        metadata = pd.read_csv(fpath, sep='|', header=None, quoting=3)
        metadata.columns = ['ID', 'Transcription', 'Normalized Transcription']
        metadata = metadata[['ID', 'Normalized Transcription']]  # we only need normalized transcription
        # metadata = metadata.sample(frac=1.0).reset_index(drop=True)  # shuffle

        if num_samples:
            metadata = metadata[:min(num_samples, metadata.shape[0])]

        return metadata

    def getWavsList(self):
        """get list of file path of .wav data"""
        wav_dir = os.path.join(self.dataDir, 'wavs')
        wavs_list = [os.path.join(wav_dir, fname + '.wav') for fname in self.metadata['ID']]
        return wavs_list

    def getOriginalText(self):
        """get original sentences"""
        return self.metadata['Normalized Transcription'].tolist()

    def getTargetSequence(self, SOS='', EOS=''):
        """get tokenized and indexed sentences """
        target_text = [SOS + txt + EOS for txt in self.metadata['Normalized Transcription']]

        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(target_text)

        target_seq = tokenizer.texts_to_sequences(target_text)
        target_seq = pad_sequences(target_seq, padding='post')

        vocab = tokenizer.word_index
        vocab['<unk>'] = 0

        vocab_rev = dict((id, char) for char, id in vocab.items())

        return target_seq, vocab, vocab_rev

    @staticmethod
    def getSpectrograms(wavs_list, n_mels, norm=True):
        """get the spectrogram corresponding to each audio"""
        spectrograms = []
        for fpath in wavs_list:
            wav, sr = librosa.load(fpath, sr=None)
            spect = librosa.feature.melspectrogram(wav, sr, n_fft=1024, n_mels=n_mels)
            spect = np.transpose(spect)
            if norm:
                mean = np.mean(spect, 1).reshape((-1, 1))
                std = np.std(spect, 1).reshape((-1, 1))
                spect = (spect - mean) / std
            spectrograms.append(spect)

        spectrograms = pad_sequences(spectrograms, padding='post')
        return spectrograms