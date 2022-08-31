# Adapted from https://github.com/intel-isl/MultiObjectiveOptimization

from PIL import Image, ImageFile

from .dataset_test import DatasetFromFolder

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
        dataset_name="cityscapes",
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

        self.data_augm_obj = DataAugmentation(
            data_augmentation,
            resize,
            image_size,
            data_transform=data_transform,
        )

    def __getitem__(self, index):
        input_img = load_img(self.input_list[index])
        # target_imgs = [self.load_img(target[index], task=self.tasks[i]) for i, target in enumerate(self.target_list)]
        segmap = None

        self.data_augm_obj.set_probabilities()

        input_img_tensor = self.data_augm_obj.apply_image_transform(input_img)[0]

        return input_img_tensor

    def __len__(self):
        return len(self.input_list)
