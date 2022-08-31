import torch.nn as nn
import functools


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_network(opt, gpu_ids=[]):

    from .deeplabv3 import deeplabv3

    netG = deeplabv3(outputs_nc=opt.outputs_nc, pretrained="None")

    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])

    return netG


def print_n_parameters_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Total number of parameters: %d" % num_params)
