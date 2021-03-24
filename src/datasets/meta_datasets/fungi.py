import os
import copy
import json
import operator
import numpy as np
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms

from src.datasets.root_paths import DATA_ROOTS


class Fungi(data.Dataset):
    NUM_CLASSES = 1394
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_fungi'], train=True, image_transforms=None):
        super().__init__()
        self.dataset = BaseFungi(
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


class BaseFungi(data.Dataset):

    def __init__(self, root=DATA_ROOTS['meta_fungi'], train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        self.data = self.load_images()

    def load_images(self):
        split = 'train' if self.train else 'val'
        with open(os.path.join(self.root, f'{split}.json')) as f:
            data_info = json.load(f)
        image_list = data_info['images']
        image_id_dict = {}
        for image in image_list:
            # assert this image_id was not previously added
            assert image['id'] not in image_id_dict
            image_id_dict[image['id']] = image

         # Add a class annotation to every image in image_id_dict.
        annotations = data_info['annotations']
        for annotation in annotations:
            # assert this images_id was not previously annotated
            assert 'class' not in image_id_dict[annotation['image_id']]
            image_id_dict[annotation['image_id']]['class'] = annotation['category_id']

        return list(image_id_dict.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.data[index]['file_name'])
        label = self.data[index]['class']
        image = Image.open(path).convert(mode='RGB')

        if self.image_transforms:
            image = self.image_transforms(image)

        return index, image, label
