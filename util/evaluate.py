# Evaluate Semantic Segmentation
from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
from glob import glob
from os.path import join
from PIL import Image
import numpy as np
from tqdm import tqdm
from util.cityscapes.labels import labels

from dataloader.dataset_bank import (
    dataset_cityscapes_get_semantics as cityscapes_semantics,
)

RESIZE_RESAMPLE_TYPE = Image.NEAREST

# Convert labels
VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
VALID_CLASSES = [
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


def get_labels():
    return [labels[c].name for c in VALID_CLASSES]


def encode_segmap(mask):
    # Not sure it is necessary
    ignore_index = 250
    class_map = dict(zip(VALID_CLASSES, range(19)))
    mask_ = np.copy(mask)
    # Put all void classes to zero
    for _voidc in VOID_CLASSES:
        mask_[mask == _voidc] = ignore_index
    for _validc in VALID_CLASSES:
        mask_[mask_ == _validc] = class_map[_validc]
    return mask_


def open_as_numpy(file, new_size=None):
    img = Image.open(file)
    if new_size is not None:
        img = img.resize(new_size, resample=RESIZE_RESAMPLE_TYPE)
    return np.asarray(img)


def plot_semantic_map(smap):
    from util.util import get_color_palette

    color_palette = np.array(get_color_palette("cityscapes"))
    color_palette = list(color_palette.reshape(-1))
    data = smap.astype(np.uint8)
    data_pil = Image.fromarray(data).convert("P")  # .putpalette(color_palette)
    data_pil.putpalette(color_palette)
    data_pil.save("/home/marcela/oi.png")


def get_pred_target_files_lp(path_to_files):
    pred_files = sorted(glob(join(path_to_files, "output/semantics/*.png")))
    path_to_dataset = "/data/datasets/public_datasets/Cityscapes/defocus-blur-experiments-none/semantics_lp_val"
    from dataloader.dataset_bank import dataset_cityscapes_get_semantics_lp

    target_files = dataset_cityscapes_get_semantics_lp(path_to_dataset)
    assert len(pred_files) == len(target_files), (
        f"Number of target files: {len(target_files)}"
        f"\n Number of prediction files: {len(pred_files)}"
    )

    return pred_files, target_files


def get_pred_target_files(path_to_files, path_to_dataset):
    target_files = cityscapes_semantics(path_to_dataset, data_split="val")
    print("Evaluating results (metrics)")
    pred_files = sorted(glob(join(path_to_files, "output/*.png")))
    assert len(pred_files) == len(target_files), (
        f"Number of target files: {len(target_files)}"
        f"\n Number of prediction files: {len(pred_files)}"
    )

    return pred_files, target_files


def evaluate(path_to_files, path_to_dataset, n_classes=19):
    if "_lp" in path_to_dataset:
        pred_files, target_files = get_pred_target_files_lp(path_to_files)
    else:
        pred_files, target_files = get_pred_target_files(path_to_files, path_to_dataset)

    global_cm = np.zeros((n_classes, n_classes))

    print("Generating metrics...")
    for pred, target in tqdm(zip(pred_files, target_files), total=len(pred_files)):
        target_np = encode_segmap(open_as_numpy(target))
        pred_np = open_as_numpy(pred, new_size=target_np.shape[::-1])
        # Resize predictions
        cm = confusion_matrix(
            target_np.ravel(), pred_np.ravel(), labels=list(range(n_classes))
        )
        plot_semantic_map(target_np)
        # plot_semantic_map(pred_np)
        global_cm += cm

    if global_cm.sum() > 0:
        overall_acc = metrics.stats_overall_accuracy(global_cm)
        average_acc, _ = metrics.stats_accuracy_per_class(global_cm)
        average_iou, iou_per_class = metrics.stats_iou_per_class(global_cm)

        print("OA: ", overall_acc)
        print("AA: ", average_acc)
        print("mIOU", average_iou)

        # IDK the order of the classes from iou_per_class
        label_names = get_labels()

        results = {"vOA": overall_acc, "vAA": average_acc, "vAIOU": average_iou}

        return results


if __name__ == "__main__":
    get_labels()
