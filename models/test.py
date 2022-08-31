import os

import numpy as np
import torch
from ipdb import set_trace as st
from PIL import Image
from tqdm import tqdm

import networks.networks as networks
from dataloader.data_loader import create_data_loader
from util.visualizer import Visualizer


def save_rgb_as_png(data, filename):
    data_np = data.data[0].cpu().float().numpy()
    data_np = np.transpose(data_np, (1, 2, 0))
    data_np = ((data_np + 1) / 2) * 255
    data_np = data_np.astype(np.uint8)
    data_pil = Image.fromarray(np.squeeze(data_np), mode="RGB")

    data_pil.save(filename)


class MTLTest:
    def name(self):
        return "Test Model for MTL"

    def initialize(self, opt, path_to_results):
        self.opt = opt
        self.save_samples_path = path_to_results
        self.opt.image_size = (
            self.opt.image_size
            if len(self.opt.image_size) == 2
            else self.opt.image_size * 2
        )
        self.gpu_ids = ""
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.netG = self.load_network()
        self.create_save_folders()

        from dataloader.dataset_cityscapes_test import DatasetCityscapes as dataset

        self.data_loader = create_data_loader(opt, dataset=dataset)

        # visualizer
        self.visualizer = Visualizer(self.opt)
        from util.util import get_color_palette

        self.opt.color_palette = np.array(get_color_palette(self.opt.dataset_name))
        self.opt.color_palette = list(self.opt.color_palette.reshape(-1))

    def load_network(self):
        # if self.opt.epoch != "latest" or self.opt.epoch != "best":
        #     self.opt.epoch = str(self.opt.epoch.zfill(4))
        checkpoint_file = os.path.join(
            self.checkpoints_path, self.opt.epoch + ".pth.tar"
        )
        if os.path.isfile(checkpoint_file):
            print(
                "Loading {} checkpoint of model {} ...".format(
                    self.opt.epoch, self.opt.name
                )
            )
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = checkpoint["epoch"]
            self.opt.net_architecture = checkpoint["arch_netG"]
            self.opt.mtl_method = checkpoint["mtl_method"]
            self.opt.n_classes = checkpoint["n_classes"]
            self.opt.outputs_nc = checkpoint["outputs_nc"][0]
            netG = self.create_network()
            pretrained_dict = checkpoint["state_dictG"]

            if "deeplab" in self.opt.net_architecture:
                for key in list(pretrained_dict.keys()):
                    if "aux" in key:
                        del pretrained_dict[key]
            netG.load_state_dict(pretrained_dict, strict=False)
            if self.opt.cuda:
                self.cuda = torch.device("cuda:0")
                netG = netG.cuda()
            self.best_val_error = checkpoint["best_pred"]

            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG
        else:
            print(
                "Couldn't find checkpoint on path: {}".format(
                    self.checkpoints_path + "/" + self.opt.epoch
                )
            )

    def get_data_loader_size(self):
        return len(self.data_loader)

    def test_and_save_target(self):
        print("Test phase using {} split.".format(self.opt.test_split))
        data_iter = iter(self.data_loader)
        self.netG.eval()
        total_iter = 0

        for it in tqdm(range(len(self.data_loader))):
            total_iter += 1
            input_cpu, targets_cpu = data_iter.next()

            input_gpu = input_cpu.to(self.cuda)

            with torch.no_grad():
                net_output_gpu = self.netG.forward(input_gpu)

            net_output_cpu = net_output_gpu[0].cpu().data[0].numpy()
            self.save_images(input_gpu, net_output_cpu, it + 1, targets_cpu)

    def test(self):
        print("Test phase using {} split.".format(self.opt.test_split))
        data_iter = iter(self.data_loader)
        self.netG.eval()
        total_iter = 0

        for it in tqdm(range(len(self.data_loader))):
            total_iter += 1
            input_cpu = data_iter.next()
            input_gpu = input_cpu.to(self.cuda)

            net_output_gpu = self.netG.forward(input_gpu)

            net_output_cpu = net_output_gpu[0].cpu().data[0].numpy()
            self.save_images(input_gpu, net_output_cpu, it + 1)

    def create_network(self):
        netG = networks.define_network(
            opt=self.opt,
            gpu_ids="",
        )
        return netG

    def save_semantics_as_png(self, data, filename):
        if "output" in filename:
            data = np.argmax(data, axis=0)
        data = data.astype(np.uint8)
        data_pil = Image.fromarray(data).convert("P")
        data_pil.putpalette(self.opt.color_palette)

        data_pil.save(filename)

    def save_as_png(self, tensor, filename):
        if "input" in filename.split("/")[-1]:
            save_rgb_as_png(data=tensor, filename=filename)
        else:
            self.save_semantics_as_png(data=tensor, filename=filename)

    def create_save_folders(self, subfolders=["rgb", "target", "output"]):
        print("Images will be saved in {}".format(self.save_samples_path))
        for sub_folder in subfolders:
            path = os.path.join(self.save_samples_path, sub_folder)
            os.system("mkdir -p {0}".format(path))
            if "rgb" not in sub_folder:
                path = os.path.join(self.save_samples_path, sub_folder)
                os.system("mkdir -p {0}".format(path))

    def save_images(self, input, net_output_cpu, index, targets=[]):
        # save other images
        self.save_as_png(
            input.data, "{}/rgb/input_{:04}.png".format(self.save_samples_path, index)
        )
        self.save_as_png(
            net_output_cpu,
            "{}/output/output_{:04}.png".format(self.save_samples_path, index),
        )

        if len(targets) > 0:
            self.save_as_png(
                targets,
                "{}/target/target_{:04}.png".format(
                    self.save_samples_path,
                    index,
                ),
            )
