"""This module contains simple helper functions """
# from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import numbers
import math
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def get_normal(depth):
    norm = np.zeros(( depth.shape[0], depth.shape[1], 3))
    dzdx = np.gradient(depth, 1, axis=0)
    dzdy = np.gradient(depth, 1, axis=1)
    norm[ :, :, 0] = -dzdx
    norm[ :, :, 1] = -dzdy
    norm[ :, :, 2] = np.ones_like(depth)
    n = np.linalg.norm(norm, axis = 2, ord=2, keepdims=True)
    norm = norm/(n + 1e-15)
    return norm

def get_normals(depth):
    norm = np.zeros(( depth.shape[0], depth.shape[1], depth.shape[2], 3))
    dzdx = np.gradient(depth, 1, axis=1)
    dzdy = np.gradient(depth, 1, axis=2)
    norm[:, :, :, 0] = -dzdx
    norm[:, :, :, 1] = -dzdy
    norm[:, :, :, 2] = np.ones_like(depth)
    n = np.linalg.norm(norm, axis = 3, ord=2, keepdims=True)
    norm = norm/(n + 1e-15)
    return norm

def logits_to_label(input):
    prob = F.softmax(input,dim=1)
    label = torch.argmax(prob, dim=1)
    return label.data.cpu().numpy()
def data_to_meters(input, opt):
    scale = opt.max_distance / 2.0
    input = input * scale + scale
    input /= 1000.0
    return input
def torch2np(input):
    return input.cpu().permute(0,2,3,1).numpy()[:,:,:,0]

def tensor2im(input, opt, isDepth = True):
    """"Converts a Tensor array into a numpy image array in meters.

    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    if not isinstance(input, np.ndarray):
        if isinstance(input, torch.Tensor):  # get the data from a variable
            tensor = input.data
        else:
            return input
        if isDepth:
            tensor = data_to_meters(tensor, opt)
            numpy = tensor.cpu().permute(0,2,3,1).numpy()[:,:,:,0]
        else:
            tensor = tensor * 127.5 + 127.5
            numpy = tensor.cpu().permute(0,2,3,1).numpy().astype(np.uint8)
    else:  # if it is a numpy array, do nothing
        numpy = input
    return numpy

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

        
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, device, dim=2):
        super(GaussianSmoothing, self).__init__()
        
        self.padding = (kernel_size - 1) // 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        self.register_buffer('weight', kernel.to(device))
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)
