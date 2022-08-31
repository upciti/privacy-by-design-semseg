from PIL import Image
import random

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_files(files_list, root):
    if len(files_list) == 0:
        from .dataset_bank import IMG_EXTENSIONS

        raise (
            RuntimeError(
                "Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
            )
        )
    else:
        print(
            "Seems like your path is ok! =) I found {} images!".format(len(files_list))
        )


def get_paths_list(root, phase, data_split):
    from .dataset_bank import dataset_cityscapes

    return dataset_cityscapes(root, data_split, phase)


def load_img(*filepaths):
    paths = []
    for path in filepaths:
        paths.append(Image.open(path))
    return paths
