import torch
import torchvision.transforms as transforms

from dataloader.dataset import DatasetFromFolder


def create_data_loader(opt, dataset=DatasetFromFolder):
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # divides float version by 255
            # normalize
        ]
    )

    if opt.test or opt.visualize:
        opt.batchSize = 1
        shuffle = False
        split = opt.test_split
        phase = "test"
    else:
        shuffle = True
        split = opt.train_split
        phase = "train"

    resize = opt.use_resize

    set_dataloader = dataset(
        opt=opt,
        root=opt.dataroot,
        phase=phase,
        data_split=split,
        data_augmentation=opt.data_augmentation,
        resize=resize,
        data_transform=data_transform,
        image_size=opt.image_size,
        output_size=opt.output_size,
        dataset_name=opt.dataset_name,
    )
    data_loader = torch.utils.data.DataLoader(
        set_dataloader,
        batch_size=opt.batchSize,
        shuffle=shuffle,
        num_workers=opt.nThreads,
        drop_last=True,
    )

    return data_loader
