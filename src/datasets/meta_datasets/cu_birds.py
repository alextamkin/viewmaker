import os
import copy
import numpy as np
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms

from src.datasets.root_paths import DATA_ROOTS


class CUBirds(data.Dataset):
    NUM_CLASSES = 200
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_cu_birds'], train=True, image_transforms=None):
        super().__init__()
        self.dataset = BaseCUBirds(
            root=root, 
            train=train,
            image_transforms=image_transforms,
        )

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        _, img_data, label = self.dataset.__getitem__(index)
        _, img2_data, _ = self.dataset.__getitem__(index)
        _, neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(), 
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


class BaseCUBirds(data.Dataset):

    def __init__(self, root=DATA_ROOTS['meta_cu_birds'], train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        # load id to image path information
        image_info_path = os.path.join(self.root, 'images.txt')
        with open(image_info_path, 'r') as f:
            image_info = [
                line.split('\n')[0].split(' ', 1) for line in f.readlines()
            ]
        image_info = dict(image_info)

        # load image to label information
        label_info_path = os.path.join(self.root, 'image_class_labels.txt')
        with open(label_info_path, 'r') as f:
            label_info = [
                line.split('\n')[0].split(' ', 1) for line in f.readlines()
            ]
        label_info = dict(label_info)

        # load train test split
        train_test_info_path = os.path.join(self.root, 'train_test_split.txt')
        with open(train_test_info_path, 'r') as f:
            train_test_info = [
                line.split('\n')[0].split(' ', 1) for line in f.readlines()
            ]
        train_test_info = dict(train_test_info)

        all_paths, all_labels = [], []
        for index, image_path in image_info.items():
            label = label_info[index]
            split = int(train_test_info[index])

            if self.train:
                if split == 1:
                    all_paths.append(image_path)
                    all_labels.append(label)
            else:
                if split == 0:
                    all_paths.append(image_path)
                    all_labels.append(label)

        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, 'images', self.paths[index])
        label = int(self.labels[index]) - 1

        image = Image.open(path).convert(mode='RGB')

        if self.image_transforms:
            image = self.image_transforms(image)

        return index, image, label
