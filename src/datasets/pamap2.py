# Wearable sensor dataset.

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchaudio.transforms import Spectrogram

import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

from src.datasets.root_paths import DATA_ROOTS

ACTIVITY_LABELS = [
    1,  # lying
    2,  # sitting
    3,  # standing
    4,  # walking
    5,  # running
    6,  # cycling
    7,  # Nordic walking
    # 9, # watching TV (optional)
    # 10, # computer work  (optional)
    # 11, # car driving (optional)
    12,  # ascending stairs
    13,  # descending stairs
    16,  # vacuum cleaning
    # 17, # ironing (optional)
    # 18, # folding laundry (optional)
    # 19, # house cleaning (optional)
    # 20, # playing soccer (optional)
    24  # rope jumping
]

FEATURE_MEANS = np.array([-10.12912214, -11.29261799, 0.67638378, 0.81824769, 0.75297834,
                          -0.35109685, 0.04085698, -0.38876906, -2.48238567, -3.41956712,
                          -3.3872513, 1.36282383, 1.55308991, 1.56087922, -10.76128503,
                          -10.35194776, -10.44513743, -10.37285293, -11.23690636, -0.20944169,
                          0.56012058,  0.807821, -1.45611818, -0.35643357, -0.25041446,
                          -2.76965766, -3.24698013, -3.85922755, 1.1442057, 1.46386916,
                          1.51837609, -11.07261072, -11.14997687, -11.13951721, -11.12178224,
                          -11.29449096, 1.94817929, 2.33591061, 1.97720141, 0.91686234,
                          1.53700002, 0.88543364, -1.64330728, -2.63160618, -2.51725697,
                          1.42671659, 1.6363767, 1.65463002, -10.68715032, -10.14333333,
                          -10.40543887, -10.2161264])
 
FEATURE_STDS = np.array([7.52822918, 6.7065013 , 3.95108152, 3.95592566, 3.42002526, 4.64231584,
                         4.44694546, 4.16510321, 3.71419447, 3.21044202, 3.59042373, 3.39598192,
                         3.24402304, 3.26736989, 4.84615018, 4.85592083, 4.75026502, 5.0713948,
                         6.76148597, 3.16121415, 4.10307909, 3.42466748, 3.91835069, 4.63133192,
                         4.12213119, 3.21565752, 3.00317751, 3.04138402, 2.96988045, 3.30489875,
                         3.05622836, 4.66155384, 4.38560602, 4.45703007, 4.35220719, 6.72132295,
                         4.49144193, 4.40899389, 3.80700876, 5.15785846, 4.82611255, 4.45424858,
                         3.65129909, 3.15463525, 3.965269  , 3.46149886, 3.22442971, 3.17674841,
                         4.71934308, 5.41595717, 4.97243856, 5.33158206])

class PAMAP2(data.Dataset):
    NUM_CLASSES = 12  # NOTE: They're not contiguous labels.
    NUM_CHANNELS = 52 # Multiple sensor readings from different parts of the body
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
        self,
        mode='train',
        sensor_transforms=None,
        root=DATA_ROOTS['pamap2'],
        examples_per_epoch=10000  # Examples are generated stochastically.
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.sensor_transforms = sensor_transforms
        self.dataset = BasePAMAP2(
            mode=mode, 
            root=root, 
            examples_per_epoch=examples_per_epoch)
    
    def transform(self, spectrogram):
        if self.sensor_transforms:
            if self.sensor_transforms == 'spectral':
                spectral_transforms = SpectrumAugmentation()
            elif self.sensor_transforms == 'spectral_noise':
                spectral_transforms = SpectrumAugmentation(noise=True)
            else:
                raise ValueError(f'Transforms {self.sensor_transforms} not implemented.')

            spectrogram = spectrogram.numpy().transpose(1, 2, 0)
            spectrogram = spectral_transforms(spectrogram)
            spectrogram = torch.tensor(spectrogram.transpose(2, 0, 1))
        elif self.sensor_transforms:
            raise ValueError(
                f'Transforms "{self.sensor_transforms}" not implemented.')
        return spectrogram

    def __getitem__(self, index):
        # pick random number
        img_data, label = self.dataset.__getitem__(index)
        subject_data = [
            index,
            self.transform(img_data).float(), 
            self.transform(img_data).float(),
            label]

        return tuple(subject_data)

    def __len__(self):
        return self.examples_per_epoch



class BasePAMAP2(data.Dataset):

    def __init__(
        self,
        mode='train',
        root=DATA_ROOTS['pamap2'],
        measurements_per_example=1000,
        examples_per_epoch=10000,
        normalize=True
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.measurements_per_example = measurements_per_example  # Measurements used to make spectrogram
        self.mode = mode
        self.subject_data = self.load_data(root)
        self.normalize = normalize

    def get_subject_ids(self, mode):
        if mode == 'train':
            nums = [1,2,3,4,7,8,9]
        elif mode == 'train_small':
            nums = [1]
        elif mode == 'val':
            nums = [5]
        elif mode == 'test':
            nums = [6]
        else:
            raise ValueError(f'mode must be one of [train, train_small, val, test]. got {mode}.')
        return nums

    def get_subject_filenames(self, mode):
        nums = self.get_subject_ids(mode)
        return [f'subject10{num}.dat' for num in nums]  # like 'subject101.dat'

    def load_data(self, root_path):
        subject_data = [] # list of data frames, one for subject
        for subject_filename in self.get_subject_filenames(self.mode):
            columns = ['timestamp', 'activity_id', 'heart_rate']
            for part in ['hand', 'chest', 'ankle']:
                for i in range(17):
                    columns.append(part + str(i))
            subj_path = os.path.join(root_path, subject_filename)
            subj_path_cache = subj_path + '.p'
            if os.path.isfile(subj_path_cache):
                print(f'Loading {subj_path_cache}')
                df = pd.read_pickle(subj_path_cache)

            else:
                df = pd.read_csv(subj_path, names=columns, sep=' ')
                df = df.interpolate()  # Interpolate out NaNs.
                print(f'Saving {subj_path_cache}')
                df.to_pickle(subj_path_cache)
            subject_data.append(df)
        return subject_data
    
    def __getitem__(self, index):
        while True:
            subject_id = np.random.randint(len(self.subject_data))
            activity_id = np.random.randint(len(ACTIVITY_LABELS))
            df = self.subject_data[subject_id]
            activity_data = df[df['activity_id'] == ACTIVITY_LABELS[activity_id]].to_numpy()
            if len(activity_data) > self.measurements_per_example: break
        start_idx = np.random.randint(len(activity_data) - self.measurements_per_example)

        # Get frame and also truncate off label and timestamp.
        # [self.measurements_per_example, 52]
        measurements = activity_data[start_idx: start_idx + self.measurements_per_example, 2:]

        # Yields spectrograms of shape [52, 32, 32]
        spectrogram_transform=Spectrogram(n_fft=64-1, hop_length=32, power=2)
        spectrogram = spectrogram_transform(torch.tensor(measurements.T))
        spectrogram = (spectrogram + 1e-6).log()
        if self.normalize:
            spectrogram = (spectrogram - FEATURE_MEANS.reshape(-1, 1, 1)) / FEATURE_STDS.reshape(-1, 1, 1)

        return spectrogram, activity_id


    def __len__(self):
        return self.examples_per_epoch


class SpectrumAugmentation(object):

    def __init__(self, noise=False):
        super().__init__()
        self.noise = noise

    def get_random_freq_mask(self):
        return nas.FrequencyMaskingAug(mask_factor=20)

    def get_random_time_mask(self):
        return nas.TimeMaskingAug(mask_factor=20)

    def __call__(self, data):
        transforms = naf.Sequential([self.get_random_freq_mask(),
                                     self.get_random_time_mask()])
        data = transforms.augment(data)
        if self.noise:
            noise_stdev = 0.25 * np.array(FEATURE_STDS).reshape(1, 1, -1)
            noise = np.random.normal(size=data.shape) * noise_stdev
            data = data + noise
        return data
