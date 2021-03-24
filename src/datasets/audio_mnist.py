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

AUDIOMNIST_MEAN = [-90.293]
AUDIOMNIST_STDEV = [11.799]
AUDIOMNIST_TRAIN_SPK = [28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                        8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]
AUDIOMNIST_VAL_SPK = [12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]
AUDIOMNIST_TEST_SPK = [26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]


class AudioMNIST(Dataset):

    def __init__(
            self,
            root=DATA_ROOTS['audio_mnist'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=True,
            max_length=150526,
            input_size=224,
            normalize_mean=AUDIOMNIST_MEAN,
            normalize_stdev=AUDIOMNIST_STDEV,
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        if train:
            speakers = AUDIOMNIST_TRAIN_SPK + AUDIOMNIST_VAL_SPK
        else:
            speakers = AUDIOMNIST_TEST_SPK
        wav_paths = []
        for spk in speakers:
            spk_paths = glob(os.path.join(root, "{:02d}".format(spk), '*.wav'))
            wav_paths.extend(spk_paths)
        self.wav_paths = wav_paths
        self.num_labels = 10
        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.max_length = max_length
        self.train = train
        self.input_size = input_size
        self.FILTER_SIZE = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label, _, _ = wav_path.rstrip(".wav").split("/")[-1].split("_")

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
