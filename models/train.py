import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict

from tqdm import tqdm
from .train_model import TrainBaseModel

from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import numpy as np


# Be able to add many loss functions
class TrainSemanticSegmentation(TrainBaseModel):
    def name(self):
        return "Semantic Segmentation Model"

    def initialize(self, opt):
        TrainBaseModel.initialize(self, opt)

        if self.opt.resume:
            self.netG, self.optimG = self.load_network()
        elif self.opt.train:
            from os.path import isdir

            if isdir(self.opt.pretrained_path) and self.opt.pretrained:
                self.netG = self.load_weights_from_pretrained_model()
            else:
                self.netG = self.create_network()
            self.optimG = self.get_optimizer(
                self.netG, self.opt.lr, weight_decay=self.opt.weightDecay
            )

        # if self.opt.display_id > 0:
        self.errors = OrderedDict()
        self.current_visuals = OrderedDict()
        self.initialize_semantics()

    def initialize_semantics(self):
        from util.util import (
            get_color_palette,
            get_dataset_semantic_weights,
            get_ignore_index,
        )

        t_index = 0
        self.opt.n_classes = self.opt.outputs_nc
        self.global_cm = np.zeros((self.opt.n_classes, self.opt.n_classes))
        self.target = self.get_variable(
            torch.LongTensor(
                self.batchSize,
                self.opt.output_nc,
                self.opt.image_size[0],
                self.opt.image_size[1],
            )
        )
        self.outG_np = None
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0
        self.opt.color_palette = get_color_palette(self.opt.dataset_name)

        weights = self.get_variable(
            torch.FloatTensor(get_dataset_semantic_weights(self.opt.dataset_name))
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weights, ignore_index=get_ignore_index(self.opt.dataset_name)
        )

    def restart_variables(self):
        self.it = 0
        self.n_iterations = 0
        self.n_images = 0
        self.e_reg = 0
        self.norm_grad_sum = 0
        self.mse_inst = 0
        self.n_images_ins = 0

    def mean_errors(self):  # called when to print current errors
        # scores
        overall_acc = metrics.stats_overall_accuracy(self.global_cm)
        average_acc, _ = metrics.stats_accuracy_per_class(self.global_cm)
        average_iou, _ = metrics.stats_iou_per_class(self.global_cm)
        self.set_current_errors(OAcc=overall_acc, AAcc=average_acc, AIoU=average_iou)

    def get_errors_semantics(self, output, target, n_classes, phase="train"):
        # e_semantics = self.cross_entropy(output, target)
        if self.old_total_iter % self.opt.print_freq == 0 or phase == "test":
            with torch.no_grad():
                target_sem_np = target.cpu().numpy()
                output_np = np.argmax(output.cpu().data.numpy(), axis=1)
                cm = confusion_matrix(
                    target_sem_np.ravel(),
                    output_np.ravel(),
                    labels=list(range(n_classes)),
                )

                if phase == "train":
                    self.global_cm += cm

                    self.set_current_visuals(
                        sem_gt=target.data.cpu().float().numpy()[0],
                        sem_out=output_np[0],
                    )
                else:
                    return cm

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        input_data = input_cpu.to(self.cuda)
        input_data.requires_grad = True
        self.set_current_visuals(input=input_data.data)
        batch_size = input_cpu.shape[0]
        self.total_iter += batch_size
        outG = self.netG.forward(input_data)
        target = target_cpu.to(self.cuda)
        self.loss_error = self.cross_entropy(outG[0], target)
        self.get_errors_semantics(outG[0], target, n_classes=self.opt.outputs_nc)

        self.optimG.zero_grad()
        self.loss_error.backward()
        self.optimG.step()

        self.n_iterations += 1

        with torch.no_grad():
            # show each loss
            self.set_current_errors_string("loss_sem", self.to_numpy(self.loss_error))

    def set_current_errors_string(self, key, value):
        self.errors.update([(key, value)])

    def set_current_errors(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.errors.update([(key, value)])

    def get_current_errors(self):
        return self.errors

    def get_current_errors_display(self):
        return self.errors

    def set_current_visuals(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.current_visuals.update([(key, value)])

    def get_current_visuals(self):
        return self.current_visuals

    def get_checkpoint(self, epoch):
        return {
            "epoch": epoch,
            "arch_netG": self.opt.net_architecture,
            "state_dictG": self.netG.state_dict(),
            "optimizerG": self.optimG,
            "best_pred": self.best_val_error,
            "mtl_method": self.opt.mtl_method,
            "outputs_nc": self.opt.outputs_nc,
            "n_classes": self.opt.n_classes,
        }

    def load_network(self):
        if self.opt.epoch is not "latest" or self.opt.epoch is not "best":
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(
            self.checkpoints_path, self.opt.epoch + ".pth.tar"
        )
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print(
                "Loading {} checkpoint of model {} ...".format(
                    self.opt.epoch, self.opt.name
                )
            )
            self.start_epoch = checkpoint["epoch"]
            self.opt.net_architecture = checkpoint["arch_netG"]
            self.opt.n_classes = checkpoint["n_classes"]
            self.opt.mtl_method = checkpoint["mtl_method"]
            netG = self.create_network()
            netG.load_state_dict(checkpoint["state_dictG"])
            optimG = checkpoint["optimizerG"]
            self.best_val_error = checkpoint["best_pred"]
            self.print_save_options()
            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG, optimG
        else:
            raise ValueError(
                "Couldn't find checkpoint on path: {}".format(
                    self.checkpoints_path + "/" + self.opt.epoch
                )
            )

    def load_weights_from_pretrained_model(self):
        epoch = "best"
        checkpoint_file = os.path.join(self.opt.pretrained_path, epoch + ".pth.tar")
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print(
                "Loading {} checkpoint of model {} ...".format(
                    epoch, self.opt.pretrained_path
                )
            )
            self.opt.net_architecture = checkpoint["arch_netG"]
            self.opt.n_classes = checkpoint["n_classes"]
            self.opt.mtl_method = checkpoint["mtl_method"]
            netG = self.create_network()
            model_dict = netG.state_dict()
            pretrained_dict = checkpoint["state_dictG"]
            model_shapes = [v.shape for k, v in model_dict.items()]
            exclude_model_dict = [
                k for k, v in pretrained_dict.items() if v.shape not in model_shapes
            ]
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and k not in exclude_model_dict
            }
            model_dict.update(pretrained_dict)
            netG.load_state_dict(model_dict)
            _epoch = checkpoint["epoch"]
            # netG.load_state_dict(checkpoint['state_dictG'])
            print("Loaded model from epoch {}".format(_epoch))
            return netG
        else:
            raise ValueError(
                "Couldn't find checkpoint on path: {}".format(
                    self.pretrained_path + "/" + epoch
                )
            )

    def to_numpy(self, data):
        return data.data.cpu().numpy()
