import os
import copy
import json
import operator
import numpy as np
from PIL import Image
from os.path import join
from itertools import chain
from scipy.io import loadmat
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms

from src.datasets.root_paths import DATA_ROOTS


class MSCOCO(data.Dataset):
    NUM_CLASSES = 80
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(
            self, 
            root=DATA_ROOTS['meta_mscoco'],
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = BaseMSCOCO(
            root=root,
            train=train,
            image_transforms=image_transforms,
        )

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)
        return index, img_data, img2_data, neg_data, label

    def __len__(self):
        return len(self.dataset)


class BaseMSCOCO(data.Dataset):
    NUM_CLASSES = 80

    def __init__(self, root=DATA_ROOTS['meta_mscoco'], train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        annotations, coco_cat_id_to_label = self.load_coco()
        paths, bboxes, labels = self.load_images(annotations, coco_cat_id_to_label)
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels
        self.targets = labels  # we sometimes query this by targets

    def load_coco(self):
        image_dir_name = ('train2017' if self.train else 'val2017')
        image_dir = join(self.root, image_dir_name)
        annotation_name = ('instances_train2017.json' if self.train else 'instances_val2017.json')
        annotation_path = join(self.root, 'annotations', annotation_name)

        with open(annotation_path, 'r') as json_file:
            annotations = json.load(json_file)
            instance_annotations = annotations['annotations']
            categories = annotations['categories']
            if len(categories) != self.NUM_CLASSES:
                raise ValueError('Total number of MSCOCO classes %d should be 80')
        
        category_ids = [cat['id'] for cat in categories]
        coco_cat_id_to_label = dict(zip(category_ids, range(len(categories))))

        return instance_annotations, coco_cat_id_to_label

    def load_images(self, annotations, coco_cat_id_to_label):
        image_dir_name = ('train2017' if self.train else 'val2017')
        image_dir = join(self.root, image_dir_name)
        all_filepaths, all_bboxes, all_labels = [], [], []
        for anno in annotations:
            image_id = anno['image_id']
            image_path = join(image_dir, '%012d.jpg' % image_id)
            bbox = anno['bbox']
            coco_class_id = anno['category_id']
            label = coco_cat_id_to_label[coco_class_id]
            all_filepaths.append(image_path)
            all_bboxes.append(bbox)
            all_labels.append(label)

        return all_filepaths, all_bboxes, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label
