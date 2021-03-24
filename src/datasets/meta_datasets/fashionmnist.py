import os
import copy
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from src.datasets.root_paths import DATA_ROOTS


class FashionMNIST(data.Dataset):
    NUM_CLASSES = 10
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_fashionmnist'], train=True, image_transforms=None):
        super().__init__()
        self.dataset = BaseFashionMNIST(
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


class BaseFashionMNIST(data.Dataset):

    def __init__(self, root=DATA_ROOTS['meta_fashionmnist'], train=True, image_transforms=None):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.image_transforms = image_transforms
        self.dataset = datasets.mnist.FashionMNIST(
            root, 
            train=train,
            download=True,
        )

    def __getitem__(self, index):
        img, target = self.dataset.data[index], int(self.dataset.targets[index])
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        return index, img, target

    def __len__(self):
        return len(self.dataset)

