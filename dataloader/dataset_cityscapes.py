# Adapted from https://github.com/intel-isl/MultiObjectiveOptimization

import torch
import torch.utils.data as data

import os
from os import listdir
from os.path import join
from PIL import Image
from ipdb import set_trace as st
import random
from math import pow
import numpy as np
from random import randint

from .dataset_utils import load_img
from .dataset import DatasetFromFolder

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from .online_data_augmentation import DataAugmentation


def load_img(filename):
    image = Image.open(filename)
    return image


class DatasetCityscapes(DatasetFromFolder):
    def __init__(
        self,
        opt,
        root,
        phase,
        data_split,
        data_augmentation,
        resize=True,
        data_transform=None,
        image_size=[256],
        output_size=0,
        dataset_name="nyu",
    ):
        DatasetFromFolder.__init__(
            self,
            opt,
            root,
            phase,
            data_split,
            data_augmentation,
            resize,
            data_transform,
            image_size,
            output_size,
            dataset_name,
        )

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]

        self.ignore_index = 250
        self.ins_ignore_value = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.data_augm_obj = DataAugmentation(
            data_augmentation,
            resize,
            image_size,
            data_transform=data_transform,
        )

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self, index):
        input_img = load_img(self.input_list[index])
        target_images = load_img(self.target_list[index])

        self.data_augm_obj.set_probabilities()

        input_img_tensor = self.data_augm_obj.apply_image_transform(input_img)[0]

        targets_np = self.data_augm_obj.apply_image_transform_no_tensor(
            target_images, resize_method=Image.BILINEAR
        )[0]
        segmap = self.encode_segmap(np.array(targets_np, dtype=np.uint8))
        tensor_array = torch.LongTensor(segmap)

        return input_img_tensor, tensor_array

    def __len__(self):
        return len(self.input_list)
