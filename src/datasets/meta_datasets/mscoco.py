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
    """Image (cropped using bounding boxes) with object labels."""
    NUM_CLASSES = 80
    FILTER_SIZE = 32
    MULTI_LABEL = False
    NUM_CHANNELS = 3

    def __init__(self, root=DATA_ROOTS['meta_mscoco'], train=True, image_transforms=None):
        super().__init__()
        self.dataset = BaseMSCOCO(
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


class BaseMSCOCO(data.Dataset):
    BOX_SCALE_RATIO = 1.2
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
        bbox = self.bboxes[index]
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image_w, image_h = image.size

        def scale_box(bbox, scale_ratio):
            x, y, w, h = bbox
            x = x - 0.5 * w * (scale_ratio - 1.0)
            y = y - 0.5 * h * (scale_ratio - 1.0)
            w = w * scale_ratio
            h = h * scale_ratio
            return [x, y, w, h]

        x, y, w, h = scale_box(bbox, self.BOX_SCALE_RATIO)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size

        if crop_width <= 0 or crop_height <= 0:
            raise ValueError('crops are not valid.')

        if self.image_transforms:
            image_crop = self.image_transforms(image_crop)

        return index, image_crop, label
