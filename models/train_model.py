# Based on cycleGAN

import os
import random
import shutil
import time
from collections import OrderedDict
from math import sqrt

import _pickle as pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

# import gc
from ipdb import set_trace as st
from torch.autograd import Variable
from tqdm import tqdm

import networks.networks as networks
from util.loss_bank import MSEScaledError
from util.visualizer import Visualizer


def _load_plot_data(filename):
    # verify if file exists
    if not os.path.isfile(filename):
        print("In _load_plot_data file {} doesnt exist.".format(filename))
        return dict
    else:
        return pickle.load(open(filename, "rb"))


class TrainBaseModel:
    def name(self):
        return "Train Model"

    def initialize(self, opt):
        self.opt = opt
        self.opt.image_size = (
            self.opt.image_size
            if len(self.opt.image_size) == 2
            else self.opt.image_size * 2
        )
        self.gpu_ids = ""
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.epoch = 0
        self.create_save_folders()

        # criterion to evaluate the val split
        self.criterion_eval = MSEScaledError()
        self.mse_scaled_error = MSEScaledError()

        self.opt.print_freq = self.opt.display_freq
        self.last_lr = self.opt.lr

        self.scheduler = None

        # visualizer
        self.visualizer = Visualizer(opt)

        if self.opt.resume and self.opt.display_id > 0:
            self.load_plot_data()
        elif opt.train:
            self.start_epoch = 1
            self.best_val_error = -999.9

        # Logfile
        self.logfile = open(os.path.join(self.checkpoints_path, "logfile.txt"), "a")
        if opt.validate:
            self.logfile_val = open(
                os.path.join(self.checkpoints_path, "logfile_val.txt"), "a"
            )

        if opt.cuda:
            self.cuda = torch.device("cuda:0")  # set externally. ToDo: set internally
            torch.cuda.manual_seed(self.opt.random_seed)

        # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
        cudnn.benchmark = self.opt.use_cudnn_benchmark
        cudnn.enabled = True

        if not opt.train and not opt.test and not opt.resume:
            raise Exception("You have to set --train or --test")

        if torch.cuda.is_available and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should run WITHOUT --cpu")
        if not torch.cuda.is_available and opt.cuda:
            raise Exception("No GPU found, run WITH --cpu")

    def set_input(self, input):
        self.input = input

    def create_network(self):
        netG = networks.define_network(
            opt=self.opt,
            gpu_ids="",
        )

        if self.opt.cuda:
            netG = netG.cuda()
        return netG

    def get_optimizer(self, network, lr, weight_decay=0.0):
        generator_params = filter(lambda p: p.requires_grad, network.parameters())
        if self.opt.optim == "Adam":
            return optim.Adam(
                generator_params,
                lr=lr,
                betas=(self.opt.beta1, 0.999),
                weight_decay=weight_decay,
            )
        elif "SGD" in self.opt.optim:
            return optim.SGD(
                generator_params, lr=lr, momentum=0.9
            )  # weight_decay=weight_decay)

    def set_current_errors_string(self, key, value):
        pass

    def set_current_errors(self, **k_dict_elements):
        pass

    def set_current_visuals(self, **k_dict_elements):
        pass

    def get_checkpoint(self, epoch):
        pass

    def train_batch(self):
        """Each method has a different implementation"""
        pass

    def display_gradients_norms(self):
        return "nothing yet"

    def get_current_errors_display(self):
        pass

    def get_regression_criterion(self):
        if self.opt.regression_loss == "L1":
            return nn.L1Loss()

    def get_variable(self, tensor, requires_grad=False):
        if self.opt.cuda:
            tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad)

    def restart_variables(self):
        self.it = 0
        self.rmse = 0
        self.n_images = 0

    def train(self, data_loader):
        self.data_loader = data_loader
        self.len_data_loader = len(self.data_loader)  # check if gonna use elsewhere
        self.total_iter = 0
        self.old_total_iter = 0
        for epoch in range(self.start_epoch, self.opt.nEpochs):
            self.epoch = epoch
            self.data_iter = iter(self.data_loader)
            self.restart_variables()
            self.pbar = tqdm(range(self.len_data_loader))
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            for self.it in self.pbar:

                self.netG.train(True)

                iter_start_time = time.time()

                self.old_total_iter += self.opt.batchSize  # depends on the iteration

                self.train_batch()

                d_time = (time.time() - iter_start_time) / self.opt.batchSize

                # print errors
                self.print_current_errors(epoch, d_time)

                # display errors
                self.display_current_results(epoch)

            # save checkpoint
            self.save_checkpoint(epoch, is_best=0)

        self.logfile.close()

        self.save_checkpoint(epoch, is_best=0, interrupt=True)

        if self.opt.validate:
            self.logfile_val.close()

    def get_next_batch(self):
        # self.it += 1 # important for GANs
        rgb_cpu, depth_cpu = self.data_iter.next()
        # depth_cpu = depth_cpu[0]
        self.input.data.resize_(rgb_cpu.size()).copy_(rgb_cpu)
        # self.target.data.resize_(depth_cpu.size()).copy_(depth_cpu)

    def apply_valid_pixels_mask(self, *data, value=0.0):
        # self.nomask_outG = data[0].data   # for displaying purposes
        mask = (data[1].data > value).to(self.cuda, dtype=torch.float32)

        masked_data = []
        for d in data:
            masked_data.append(d * mask)

        return masked_data, mask.sum()

    # UPDATE LEARNING RATE

    def polynomial_learning_rate(self, optim):
        """
        From Tensorflow: https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay
        """
        end_lr = self.opt.lr * 1e-2
        decay_steps = 3000000000  # like tensorflow example
        power = 0.1

        global_it = self.total_iter

        if global_it > decay_steps:
            self.set_current_errors(lastlr=self.last_lr)
            return optim

        # new_lr = self.opt.lr * (1 - global_it / decay_steps) ** power
        self.global_step = min(global_it, decay_steps)
        new_lr = (self.last_lr - end_lr) * (1 - (global_it / decay_steps)) ** (
            power
        ) + end_lr
        for param_group in optim.param_groups:
            # Adam changes learning rates
            param_group["lr"] = (self.last_lr - end_lr) * (
                1 - (global_it / decay_steps)
            ) ** (power) + end_lr
        self.last_lr = new_lr
        self.set_current_errors(lastlr=self.last_lr)
        return optim

    # CONTROL FUNCTIONS OF THE ARCHITECTURE

    def _get_plot_data_filename(self, phase):
        return os.path.join(
            self.checkpoints_path,
            "plot_data" + ("" if phase == "train" else "_" + phase) + ".p",
        )

    def save_static_plot_image():
        return None

    def save_interactive_plot_image():
        return None

    def _save_plot_data(self, plot_data, filename):
        # save
        pickle.dump(plot_data, open(filename, "wb"))

    def save_plot_data(self):
        self._save_plot_data(
            self.visualizer.plot_data, self._get_plot_data_filename("train")
        )
        if self.opt.validate and self.total_iter > self.opt.val_freq:
            self._save_plot_data(
                self.visualizer.plot_data_val, self._get_plot_data_filename("val")
            )

    def load_plot_data(self):
        self.visualizer.plot_data = _load_plot_data(
            self._get_plot_data_filename("train")
        )
        if self.opt.validate:
            self.visualizer.plot_data_val = _load_plot_data(
                self._get_plot_data_filename("val")
            )

    def save_checkpoint(self, epoch, is_best, interrupt=False):
        if epoch % self.opt.save_checkpoint_freq == 0 or is_best or interrupt:
            if self.opt.display_id > 0:
                self.save_plot_data()

            checkpoint = self.get_checkpoint(epoch)
            checkpoint_filename = "{}/{:04}.pth.tar".format(
                self.checkpoints_path, epoch
            )
            if interrupt:
                checkpoint_filename = "{}/interrupt_{:04}.pth.tar".format(
                    self.checkpoints_path, epoch
                )
            self._save_checkpoint(
                checkpoint, is_best=is_best, filename=checkpoint_filename
            )  # standart is_best=0 here cause we didn' evaluate on validation data
            # save plot data as well

    def get_epoch(self):
        return self.epoch

    def _save_checkpoint(self, state, is_best, filename):
        print("Saving checkpoint from model {}...".format(self.opt.name))
        # uncomment next 2 lines if we still want per epoch
        torch.save(state, filename)
        shutil.copyfile(
            filename, os.path.join(os.path.dirname(filename), "latest.pth.tar")
        )

        # comment next 2 lines if necessary if using last two lines
        # filename = os.path.join(self.checkpoints_path, 'latest.pth.tar')
        # torch.save(state, os.path.join(self.checkpoints_path, 'latest.pth.tar'))

        if is_best:
            shutil.copyfile(
                filename, os.path.join(self.checkpoints_path, "best.pth.tar")
            )

    def create_save_folders(self):
        if self.opt.train:
            os.system("mkdir -p {0}".format(self.checkpoints_path))
        # if self.opt.save_samples:
        #     subfolders = ['input', 'target', 'results', 'output']
        #     self.save_samples_path = os.path.join('results/train_results/', self.opt.name)
        #     for subfolder in subfolders:
        #         path = os.path.join(self.save_samples_path, subfolder)
        #         os.system('mkdir -p {0}'.format(path))
        #     if self.opt.test:
        #         self.save_samples_path = os.path.join('results/test_results/', self.opt.name)
        #         self.save_samples_path = os.path.join(self.save_samples_path, self.opt.epoch)
        #         for subfolder in subfolders:
        #             path = os.path.join(self.save_samples_path, subfolder)
        #             os.system('mkdir -p {0}'.format(path))

    def print_save_options(self):
        options_file = open(os.path.join(self.checkpoints_path, "options.txt"), "w")
        args = dict(
            (arg, getattr(self.opt, arg))
            for arg in dir(self.opt)
            if not arg.startswith("_")
        )
        print("---Options---")
        for k, v in sorted(args.items()):
            option = "{}: {}".format(k, v)
            # print options
            print(option)
            # save options in file
            options_file.write(option + "\n")

        options_file.close()

    def mean_errors(self):
        pass

    def get_current_errors(self):
        pass

    def print_current_errors(self, epoch, d_time):
        if self.old_total_iter % self.opt.print_freq == 0:
            self.mean_errors()
            errors = self.get_current_errors()
            message = self.visualizer.print_errors(
                errors, epoch, self.it, self.len_data_loader, d_time
            )

            self.pbar.set_description(message)

    def get_current_visuals(self):
        pass

    def display_current_results(self, epoch):
        if self.opt.display_id > 0 and self.old_total_iter % self.opt.display_freq == 0:
            errors = self.get_current_errors_display()
            self.visualizer.display_errors(
                errors, epoch, float(self.it) / self.len_data_loader
            )

            visuals = self.get_current_visuals()

            self.visualizer.display_images(visuals, epoch)

            # save printed errors to logfile
            self.visualizer.save_errors_file(self.logfile)
            # self.save_plot_data()

    def get_mask(self, data, value=0.0):
        return target.data > 0.0

    def get_padding(self, dim):
        final_dim = (dim // 32 + 1) * 32
        return final_dim - dim

    def get_padding_image(self, img):
        # get tensor dimensions
        h, w = img.size()[2:]
        w_pad, h_pad = self.get_padding(w), self.get_padding(h)

        pwr = w_pad // 2
        pwl = w_pad - pwr
        phb = h_pad // 2
        phu = h_pad - phb

        # pwl, pwr, phu, phb
        return (pwl, pwr, phu, phb)

    def adjust_learning_rate(self, initial_lr, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = initial_lr * (0.1 ** (epoch // self.opt.niter_decay))
        if epoch % self.opt.niter_decay == 0:
            print("LEARNING RATE DECAY HERE: lr = {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
