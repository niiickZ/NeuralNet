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
        self.metadata = self.readMetadata(dataDir, num_samples)

        wav_dir = os.path.join(dataDir, 'wavs')
        self.wavs_list = [os.path.join(wav_dir, fname + '.wav') for fname in self.metadata['ID']]

    def readMetadata(self, dataDir, num_samples):
        """Read meta data"""
        fpath = os.path.join(dataDir, 'metadata.csv')
        metadata = pd.read_csv(fpath, sep='|', header=None, quoting=3)
        metadata.columns = ['ID', 'Transcription', 'Normalized Transcription']
        metadata = metadata[['ID', 'Normalized Transcription']]  # we only need normalized transcription
        # metadata = metadata.sample(frac=1.0).reset_index(drop=True)  # shuffle

        if num_samples:
            metadata = metadata[:min(num_samples, metadata.shape[0])]

        return metadata

    def getWavsList(self):
        """get list of file path of .wav data"""
        return self.wavs_list

    def getOriginalText(self):
        """get original sentences"""
        return self.metadata['Normalized Transcription'].tolist()

    def getTargetSequence(self):
        """get tokenized and indexed sentences """
        target_text = self.metadata['Normalized Transcription'].tolist()

        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(target_text)

        target_seq = tokenizer.texts_to_sequences(target_text)
        target_seq = pad_sequences(target_seq, padding='post')

        vocab = tokenizer.word_index
        vocab['<unk>'] = 0

        vocab_rev = dict((id, char) for char, id in vocab.items())

        return target_seq, vocab, vocab_rev

    @staticmethod
    def getSpectrograms(wavs_list, mxlen, n_mels):
        """get the spectrogram corresponding to each audio"""
        spectrograms = []
        for fpath in wavs_list:
            wav, sr = librosa.load(fpath, sr=None)
            spect = librosa.feature.melspectrogram(wav, sr, n_fft=1024, n_mels=n_mels)
            spect = np.transpose(spect)
            spectrograms.append(spect)

        spectrograms = pad_sequences(spectrograms, maxlen=mxlen, padding='post', truncating='post')
        return spectrograms