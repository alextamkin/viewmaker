import os
import torch
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
from glob import glob
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize

from torch.utils.data import Dataset
from nlpaug.augmenter.audio import AudioAugmenter
from src.datasets.librispeech import WavformAugmentation, SpectrumAugmentation
from src.datasets.root_paths import DATA_ROOTS

FLUENTSPEECH_MEAN = [-31.809]
FLUENTSPEECH_STDEV = [13.127]

FLUENTSPEECH_ACTIONS = [
    'change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring'
]
FLUENTSPEECH_OBJECTS = [
    'none', 'music', 'lights', 'volume', 'heat', 'lamp', 'newspaper', 'juice', 'socks', 'shoes', 'Chinese', 'Korean', 'English', 'German'
]
FLUENTSPEECH_LOCATIONS = [
    'none', 'kitchen', 'bedroom', 'washroom'
]


class FluentSpeechCommands(Dataset):

    def __init__(
            self,
            label_type='action',
            root=DATA_ROOTS['fluent_speech'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            max_length=150526,
            input_size=224,
            normalize_mean=FLUENTSPEECH_MEAN,
            normalize_stdev=FLUENTSPEECH_STDEV,
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        assert label_type in ['action', 'object', 'location']

        if train:
            train_path = os.path.join(root, 'data', 'train_data.csv')
            val_path = os.path.join(root, 'data', 'valid_data.csv')
            train_data = pd.read_csv(train_path)
            train_paths = list(train_data['path'])
            train_labels = list(train_data[label_type])
            val_data = pd.read_csv(val_path)
            val_paths = list(val_data['path'])
            val_labels = list(val_data[label_type])
            wav_paths = train_paths + val_paths
            labels = train_labels + val_labels
        else:
            test_path = os.path.join(root, 'data', 'test_data.csv')
            test_data = pd.read_csv(test_path)
            wav_paths = list(test_data['path'])
            labels = list(test_data[label_type])
       
        if label_type == 'action':
            self.num_labels = len(FLUENTSPEECH_ACTIONS)
        elif label_type == 'object':
            self.num_labels = len(FLUENTSPEECH_OBJECTS)
        elif label_type == 'location':
            self.num_labels = len(FLUENTSPEECH_LOCATIONS)

        self.root = root
        self.label_type = label_type
        self.wav_paths = wav_paths
        self.labels = labels
        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.max_length = max_length
        self.train = train
        self.input_size = input_size
        self.FILTER_SIZE = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev

    def __getitem__(self, index):
        wav_name = self.wav_paths[index]
        wav_path = os.path.join(self.root, wav_name)

        label = self.labels[index]

        if self.label_type == 'action':
            label = FLUENTSPEECH_ACTIONS.index(label)
        elif self.label_type == 'object':
            label = FLUENTSPEECH_OBJECTS.index(label)
        elif self.label_type == 'location':
            label = FLUENTSPEECH_LOCATIONS.index(label)

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        if self.wavform_transforms:
            transforms = WavformAugmentation(sample_rate)
            wavform = transforms(wavform)

        # pad to 150k frames
        if len(wavform) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:self.max_length] if flip else 
                      wavform[-self.max_length:])
        else:
            padded = np.zeros(self.max_length)
            padded[:len(wavform)] = wavform  # pad w/ silence

        hop_length_dict = {224: 672, 112: 1344, 64: 2360, 32: 4800}
        spectrum = librosa.feature.melspectrogram(
            padded,
            sample_rate,
            hop_length=hop_length_dict[self.input_size],
            n_mels=self.input_size,
        )

        if self.spectral_transforms:  # apply time and frequency masks
            transforms = SpectrumAugmentation()
            spectrum = transforms(spectrum)

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        if self.spectral_transforms:  # apply noise on spectral
            noise_stdev = 0.25 * self.normalize_stdev[0]
            noise = torch.randn_like(spectrum) * noise_stdev
            spectrum = spectrum + noise

        normalize = Normalize(self.normalize_mean, self.normalize_stdev)
        spectrum = normalize(spectrum)

        return index, spectrum, int(label)

    def __len__(self):
        return len(self.wav_paths)
