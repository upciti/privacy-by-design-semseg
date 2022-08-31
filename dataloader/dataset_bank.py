from os import listdir
from os.path import join
from ipdb import set_trace as st
import glob
import sys

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def dataset_cityscapes_get_rgb(root, data_split):
    path = join(root, "leftImg8bit", data_split)
    city = "*"
    return sorted(glob.glob(join(path, city, "*.png")))


def dataset_cityscapes_get_defocused_rgb(root, data_split, focus_prefix=""):
    path = join(
        root,
        "defocus-blur-experiments" + focus_prefix,
        "defocused_leftImg8bit",
        data_split,
    )
    city = "*"
    print(path)
    return sorted(glob.glob(join(path, city, "*.png")))


def dataset_cityscapes_get_defocused_rgb_focal_plane(root, data_split, focus_prefix=""):
    path = join(root, "defocus-blur-experiments" + focus_prefix, "lp_val")
    print(path)
    return sorted(glob.glob(join(path, "*.png")))


def dataset_cityscapes_get_disparity(root, data_split):
    path = join(root, "disparity", data_split)
    city = "*"
    return sorted(glob.glob(join(path, city, "*.png")))


def dataset_cityscapes_get_instance(root, data_split):
    path = join(root, "gtFine_trainvaltest/gtFine", data_split)
    city = "*"
    return sorted(glob.glob(join(path, city, "*instanceIds.png")))


def dataset_cityscapes_get_semantics(root, data_split):
    path = join(root, "gtFine_trainvaltest/gtFine", data_split)
    city = "*"
    return sorted(glob.glob(join(path, city, "*labelIds.png")))


def dataset_cityscapes_get_semantics_lp(root):
    path = join(root, "*labelIds.png")
    return sorted(glob.glob(path))


def dataset_cityscapes(root, data_split, phase):
    if "defocus_lp" in root:
        focus_prefix = root.split("/")[-1].replace("defocus_lp", "")
        root = root.split("/")[:-1]
        root = join(*root)
        rgb_images = dataset_cityscapes_get_defocused_rgb_focal_plane(
            root, data_split, focus_prefix
        )
    elif "defocus" in root:
        focus_prefix = root.split("/")[-1].replace("defocus", "")
        root = root.split("/")[:-1]
        root = join(*root)
        rgb_images = dataset_cityscapes_get_defocused_rgb(
            root, data_split, focus_prefix
        )
    else:
        rgb_images = dataset_cityscapes_get_rgb(root, data_split)
    if phase == "test":
        return rgb_images

    return rgb_images, dataset_cityscapes_get_semantics(root, data_split)
