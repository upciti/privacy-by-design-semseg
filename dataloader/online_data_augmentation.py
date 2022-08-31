import os
import random
from math import pow
from os import listdir
from os.path import join
from random import randint

import imgaug.augmenters as iaa
import numpy as np
import torch.utils.data as data
from ipdb import set_trace as st
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rotate_image(img, rotation):
    return img.rotate(rotation, resample=Image.BILINEAR)


def str2bool(values):
    return [v.lower() in ("true", "t") for v in values]


def img_to_grayscale(img):
    img = np.asarray(ImageOps.grayscale(img))
    img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    return Image.fromarray(np.uint8(img))


class DataAugmentation:
    def __init__(
        self,
        data_augmentation,
        resize,
        image_size,
        mean_rotation=0,
        max_rotation=5.0,
        data_transform=None,
    ):
        print(str2bool(data_augmentation))
        (
            self.hflip,
            self.gray,
        ) = str2bool(data_augmentation)

        self.data_transform = data_transform
        self.resize = resize
        self.image_size = image_size
        self.mean_rotation = mean_rotation
        self.max_rotation = max_rotation

        print("\nData Augmentation")
        print("Hflip: {}".format(self.hflip))
        print("Gray: {}".format(self.gray))

    def set_probabilities(self):
        self.prob_hflip = random.random()
        self.prob_rotation = np.random.normal(
            self.mean_rotation, self.max_rotation / 3.0
        )
        self.gray_probability = 1  # random.randint(0, 1)

        global state
        state = random.getstate()

    def apply_image_transform(self, *arrays):
        import torch

        results = []

        for i, img in enumerate(arrays):
            if self.gray and img.mode == "RGB" and self.gray_probability:
                img = img_to_grayscale(img)
            if self.hflip and self.prob_hflip < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.resize:
                resize_method = Image.BILINEAR  # Image.ANTIALIAS
                img = img.resize(
                    (self.image_size[0], self.image_size[1]), resize_method
                )
            # To tensor
            if img.mode == "P" or img.mode != "RGB":  # label
                img_tensor = torch.LongTensor(np.array(img, dtype=np.int64))
            elif img.mode == "RGB":
                img_tensor = self.data_transform(img)
                img_tensor = (img_tensor * 2) - 1
            results.append(img_tensor)
        return results

    def apply_image_transform_no_tensor(self, *arrays, resize_method=Image.BILINEAR):
        results = []

        for i, img in enumerate(arrays):
            # DA: Data augmentation
            if self.hflip and self.prob_hflip < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.resize:
                img = img.resize(
                    (self.image_size[0], self.image_size[1]), resize_method
                )
            results.append(img)
        return results
