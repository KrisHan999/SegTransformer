import torch
import torch.nn.functional as F
import scipy.ndimage.filters as fi
import numpy as np

def gkern2(kernlen, nsig, device):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    filter = fi.gaussian_filter(inp, nsig)
    filter = torch.tensor(filter, device=device).unsqueeze(0).unsqueeze(0).float()
    return filter


def skern2(kernlen, device):
    """Returns a 2D square kernel array."""
    # create nxn zeros
    inp = np.ones((kernlen, kernlen), dtype=np.float)
    filter = inp / inp.sum()
    filter = torch.tensor(filter, device=device).unsqueeze(0).unsqueeze(0).float()
    return filter


def ckern2(kernlen, device):
    """Returns a 2D circle kernel array. plot the kernel first"""
    inp = np.zeros((kernlen, kernlen), dtype=np.float)
    for i in range(kernlen):
        for j in range(kernlen):
            if np.sqrt((i-kernlen//2)**2+(j-kernlen//2)**2) <= kernlen//2:
                inp[i][j] = 1
    filter = inp / inp.sum()
    filter = torch.tensor(filter, device=device).unsqueeze(0).unsqueeze(0).float()
    return filter


def generate_volume_pyramid(volume, n_layer):
    """
        generate a pyramid of images with different resolutions.
    :param volume: [N, C, H, W]
    :param n_layer:
    :return:
    """

    volume_pyramid = []
    N, C, H, W = volume.shape
    device = volume.device
    filter = gkern2(3, 1, device)
    volume_pyramid.append(volume)
    temp_volume = volume
    for i in range(1, n_layer):
        new_volume = torch.zeros(N, C, H//2**i, W//2**i, device=device)
        for n in range(N):
            for c in range(C):
                new_volume[n][c] = F.max_pool2d(F.conv2d(temp_volume[n][c][None, None, ...], filter, stride=1, padding=1).gt(0).float(), 2)
        volume_pyramid.append(new_volume)
        temp_volume = new_volume
    return volume_pyramid


def generate_filtered_volume(volume, filter, stride=1, padding=1):
    """

    :param volume: [N, C, H, W]
    :param filter: [1, 1, kernel_size, kernel_size]
    :return: [N, C, H, W]
    """
    N, C, H, W = volume.shape
    new_volume = torch.zeros_like(volume, device=volume.device)
    for n in range(N):
        for c in range(C):
            new_volume[n][c] = F.conv2d(volume[n][c][None, None, ...], filter, stride=stride, padding=padding)
    return new_volume



