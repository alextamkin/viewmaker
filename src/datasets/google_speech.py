import os
import torch
import random
import librosa
import torchaudio
import numpy as np
from glob import glob
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize

from torch.utils.data import Dataset
from nlpaug.augmenter.audio import AudioAugmenter
from src.datasets.librispeech import WavformAugmentation, SpectrumAugmentation
from src.datasets.root_paths import DATA_ROOTS

GOOGLESPEECH_MEAN = [-46.847]
GOOGLESPEECH_STDEV = [19.151]
GOOGLESPEECH_LABELS = ['eight', 'right', 'happy', 'three', 'yes', 'up', 'no', 'stop', 'on', 'four', 'nine', 
                       'zero', 'down', 'go', 'six', 'two', 'left', 'five', 'off', 'seven', 'one',
                       'cat', 'bird', 'marvin', 'wow', 'tree', 'dog', 'sheila', 'bed', 'house']


class GoogleSpeechCommands(Dataset):

    def __init__(
            self,
            root=DATA_ROOTS['google_speech'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            max_length=150526,
            input_size=224,
            normalize_mean=GOOGLESPEECH_MEAN,
            normalize_stdev=GOOGLESPEECH_STDEV,
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        if train:
            train_paths = open(os.path.join(root, 'training_list.txt'), 'r').readlines()
            val_paths = open(os.path.join(root, 'validation_list.txt'), 'r').readlines()
            wav_paths = train_paths + val_paths
        else:
            test_paths = open(os.path.join(root, 'testing_list.txt'), 'r').readlines()
            wav_paths = test_paths
        
        wav_paths = [path.strip() for path in wav_paths]

        self.root = root
        self.num_labels = len(GOOGLESPEECH_LABELS)
        self.wav_paths = wav_paths
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
        label_name = wav_name.split('/')[0].lower()
        label = GOOGLESPEECH_LABELS.index(label_name)
        wav_path = os.path.join(self.root, wav_name)

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
