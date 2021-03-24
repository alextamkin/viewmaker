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


class Aircraft(data.Dataset):
    NUM_CLASSES = 100
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_aircraft'], train=True, image_transforms=None, seed=42):
        super().__init__()
        self.dataset = BaseAircraft(
            root=root, 
            train=train,
            image_transforms=image_transforms,
            seed=seed,
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


class BaseAircraft(data.Dataset):

    def __init__(self, root=DATA_ROOTS['meta_aircraft'], train=True, image_transforms=None, seed=42):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        paths, bboxes, labels = self.load_images()
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels

    def load_images(self):
        split = 'trainval' if self.train else 'test'
        variant_path = os.path.join(self.root, 'data', f'images_variant_{split}.txt')
        with open(variant_path, 'r') as f:
            names_to_variants = [
                line.split('\n')[0].split(' ', 1) for line in f.readlines()
            ]
        names_to_variants = dict(names_to_variants)

        # Build mapping from variant to filenames. "Variant" refers to the aircraft
        # model variant (e.g., A330-200) and is used as the class name in the
        # dataset. The position of the class name in the concatenated list of
        # training, validation, and test class name constitutes its class ID.
        variants_to_names = defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)

        names_to_bboxes = self.get_bounding_boxes()

        variants = sorted(list(set(variants_to_names.keys())))
        split_files, split_labels, split_bboxes = [], [], []

        for variant_id, variant in enumerate(variants):
            class_files = [join(self.root, 'data', 'images', f'{filename}.jpg')
                            for filename in sorted(variants_to_names[variant])]
            bboxes = [names_to_bboxes[name]
                        for name in sorted(variants_to_names[variant])]
            labels = list([variant_id] * len(class_files))

            split_files += class_files
            split_labels += labels
            split_bboxes += bboxes

        return split_files, split_bboxes, split_labels

    def get_bounding_boxes(self):
        bboxes_path = os.path.join(self.root, 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [
                line.split('\n')[0].split(' ') for line in f.readlines()
            ]
            names_to_bboxes = dict(
                (name, list(map(int, (xmin, ymin, xmax, ymax))))
                for name, xmin, ymin, xmax, ymax in names_to_bboxes)

        return names_to_bboxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bbox = tuple(self.bboxes[index])
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = image.crop(bbox)

        if self.image_transforms:
            image = self.image_transforms(image)

        return index, image, label

