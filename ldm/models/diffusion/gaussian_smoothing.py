"""Gaussian smoothing layer (used by BoxDiff attention loss)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianSmoothing(nn.Module):
    """Apply Gaussian smoothing on a tensor.

    Args:
        channels  : number of input channels
        kernel_size: size of the square Gaussian kernel
        sigma      : standard deviation of the Gaussian kernel
        dim        : number of spatial dimensions (2 for images)
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, (int, float)):
            sigma = [sigma] * dim

        kernel = 1.0
        meshgrids = torch.meshgrid(
            [torch.arange(k, dtype=torch.float32) for k in kernel_size],
            indexing='ij',
        )
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.0
            kernel *= torch.exp(-((grid - mean) ** 2) / (2 * std ** 2))

        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f'Only 1, 2 and 3 dimensions are supported. Got {dim}.')

    def forward(self, x):
        return self.conv(x, weight=self.weight.to(x.dtype), groups=self.groups)
