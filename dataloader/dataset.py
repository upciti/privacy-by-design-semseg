import torch.utils.data as data
from PIL import ImageFile

from .dataset_utils import check_files, get_paths_list, load_img

ImageFile.LOAD_TRUNCATED_IMAGES = True
from .online_data_augmentation import DataAugmentation


class DatasetFromFolder(data.Dataset):
    def __init__(
        self,
        opt=None,
        root=None,
        phase=None,
        data_split=None,
        data_augmentation=None,
        resize=True,
        data_transform=None,
        image_size=[256],
        output_size=0,
        dataset_name="",
    ):
        super(DatasetFromFolder, self).__init__()
        self.input_list, self.target_list = get_paths_list(root, phase, data_split)

        check_files(self.input_list, root)

        self.data_transform = data_transform
        self.image_size = image_size if len(image_size) == 2 else image_size * 2
        self.dataset_name = dataset_name

        if output_size == 0:
            self.output_size = self.image_size
        else:
            self.output_size = output_size if len(output_size) == 2 else output_size * 2

        self.data_augmentation = data_augmentation
        self.resize = resize
        self.state = 0

        self.phase = phase
        self.data_augm_obj = DataAugmentation(
            data_augmentation,
            resize,
            self.image_size,
            data_transform=self.data_transform,
        )
        self.data_augm_obj.set_probabilities()

    def __getitem__(self, index):
        # not used on cityscapes
        input_img = load_img(self.input_list[index])[0]
        target_imgs = [load_img(target[index])[0] for target in self.target_list[index]]

        self.data_augm_obj.set_probabilities()

        input_img_tensor = self.data_augm_obj.apply_image_transform(input_img)[0]

        targets_tensor = [
            self.data_augm_obj.apply_image_transform(target)[0]
            for i, target in enumerate(target_imgs)
        ]
        return input_img_tensor, targets_tensor

    def __len__(self):
        return len(self.input_list)
