import random
from os.path import join

import numpy as np
import torch

from dataloader.data_loader import create_data_loader
from options.arguments import TrainTestOptions as ArgumentOptions


def make_deterministic(seed):
    # Built-in Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Load options
    opt = ArgumentOptions().parse()

    make_deterministic(opt.random_seed)

    if opt.cuda:
        cuda = torch.device("cuda:0")  # set externally. ToDo: set internally

    # train model
    if opt.train or opt.resume:
        from models.train import TrainSemanticSegmentation as Model

        # Initialize model
        model = Model()
        model.initialize(opt)
        from dataloader.dataset_cityscapes import DatasetCityscapes as dataset

        # Create dataloaders
        data_loader = create_data_loader(opt, dataset=dataset)

        try:
            model.train(data_loader)

        except KeyboardInterrupt:
            print("Stopping early. Saving network...")
            # TODO implement saving network here and others
            epoch = model.get_epoch()
            model.save_checkpoint(epoch, is_best=False, interrupt=True)
            exit()
        except BaseException as e:
            print("Another reason: {}".format(e))
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # print(traceback.format_exc(exc_traceback))
            epoch = model.get_epoch()
            model.save_checkpoint(epoch, is_best=False, interrupt=True)

    elif opt.test:
        from models.test import MTLTest as Model

        path_to_results = join("results", opt.name, opt.epoch, opt.input_type)

        print(path_to_results)
        if opt.test_only or opt.test_metrics:
            # Only generates predictions
            model = Model()
            model.initialize(opt, path_to_results)
            model.test()
            del model
        if opt.evaluate_only or opt.test_metrics:
            # expects to have the inference results
            from util.evaluate import evaluate

            results = evaluate(path_to_results, opt.dataroot, n_classes=opt.n_classes)
