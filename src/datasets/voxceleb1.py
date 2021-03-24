import os
import torch
import random
import librosa
import torchaudio
import numpy as np
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from collections import defaultdict
from src.datasets.librispeech import WavformAugmentation, SpectrumAugmentation
from src.datasets.root_paths import DATA_ROOTS

VOX_CELEB_MEAN = [-37.075]
VOX_CELEB_STDEV = [19.776]


class VoxCeleb1(Dataset):

    def __init__(
            self,
            root=DATA_ROOTS['voxceleb1'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            max_length=150526,
            input_size=224,
            normalize_mean=VOX_CELEB_MEAN,
            normalize_stdev=VOX_CELEB_STDEV,
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        self.root = root
        wav_paths, speaker_strs = self.get_split(train)
        # change speaker_strs to integers 
        unique_speakers = sorted(set(speaker_strs))
        speaker_id_map = dict(zip(unique_speakers, range(len(unique_speakers))))
        speaker_ids = [speaker_id_map[sp] for sp in speaker_strs]

        self.train = train
        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.wav_paths = wav_paths
        self.max_length = max_length
        self.speaker_ids = speaker_ids
        self.num_unique_speakers = len(unique_speakers)
        self.num_labels = len(unique_speakers)
        self.input_size = input_size
        self.FILTER_SIZE = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev

    def get_split(self, train=True):
        split_file = os.path.join(self.root, 'iden_split.txt')
        with open(split_file, 'r') as fp:
            splits = fp.readlines()

        paths = defaultdict(lambda: [])
        for split in splits:
            spl, path = split.strip().split(' ')
            paths[spl].append(path)

        train_paths = paths['1'] + paths['2']
        test_paths  = paths['3']

        train_speaker_ids = [p.split('/')[0] for p in train_paths]
        test_speaker_ids = [p.split('/')[0] for p in test_paths]

        if train:
            return train_paths, train_speaker_ids
        else:
            return test_paths, test_speaker_ids

    def __getitem__(self, index):
        wav_path = os.path.join(self.root, 'wav', self.wav_paths[index])
        speaker_id = self.speaker_ids[index]
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

        return index, spectrum, speaker_id

    def __len__(self):
        return len(self.wav_paths)
