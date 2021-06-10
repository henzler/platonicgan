import numpy as np
import torch
import math
import cv2
import torch.nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from skimage.measure import compare_ssim as ssim


def pytorch_to_numpy(array, is_batch=True, flip=True):
    array = array.detach().cpu().numpy()

    if flip:
        source = 1 if is_batch else 0
        dest = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    return array


def numpy_to_pytorch(array, is_batch=False, flip=True):
    if flip:
        dest = 1 if is_batch else 0
        source = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    array = torch.from_numpy(array)
    array = array.float()

    return array


def convert_to_int(array):
    array *= 255

    if type(array).__module__ == 'numpy':
        return array.astype(np.uint8)

    elif type(array).__module__ == 'torch':
        return array.byte()
    else:
        raise NotImplementedError


def convert_to_float(array):
    if type(array).__module__ == 'numpy':
        return array.astype(np.float32) / 255.0

    elif type(array).__module__ == 'torch':
        return array.float() / 255.0
    else:
        raise NotImplementedError


def nth_root(val, n):
    ret = int(val ** (1. / n))
    return ret + 1 if (ret + 1) ** n == val else ret


def normalize(array):
    array = np.float32(array)
    array = (array - array.min()) / max(1, (array.max() - array.min()))  # [0, 1]
    return array


def generate_z(param, size, single=False):
    if single:
        return torch.Tensor(1, size).uniform_(-1.0, 1.0).to(param.device)
    else:
        return torch.Tensor(param.training.batch_size, size).uniform_(-1.0, 1.0).to(param.device)


def polar_to_cartesian(array, r=1.0):
    theta = array[..., 0]
    u = array[..., 1]

    x = torch.sqrt(1 - u * u) * torch.sin(theta)
    y = u
    z = torch.sqrt(1 - u * u) * torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


def get_pytorch_grid_coords_3d(z_size, y_size, x_size, coord_dim=-1):
    steps_x = torch.linspace(-1.0, 1.0, x_size)
    steps_y = torch.linspace(-1.0, 1.0, y_size)
    steps_z = torch.linspace(-1.0, 1.0, z_size)
    z, y, x = torch.meshgrid(steps_z, steps_y, steps_x)
    coords = torch.stack([x, y, z], dim=coord_dim)
    return coords


def get_graphics_grid_coords_3d(z_size, y_size, x_size, coord_dim=-1):
    steps_x = torch.linspace(-1.0, 1.0, x_size)
    steps_y = torch.linspace(1.0, -1.0, y_size)
    steps_z = torch.linspace(1.0, -1.0, z_size)
    z, y, x = torch.meshgrid(steps_z, steps_y, steps_x)
    coords = torch.stack([x, y, z], dim=coord_dim)
    return coords


def reflect_vector(vector, normal=[-1.0, 0.0, 0.0]):
    normal = torch.tensor(normal).repeat(vector.size(0), 1)

    if vector.is_cuda:
        normal = normal.to('cuda')

    return vector - 2.0 * torch.bmm(normal.unsqueeze(1), vector.unsqueeze(2))[:, :, 0] * normal


def metric_iou(output, target):
    e = 1e-8

    output = output.copy()
    target = target.copy()

    output[output > e] = 1
    target[target > e] = 1

    target[target <= e] = 0
    target[target <= e] = 0

    output = np.array(output, dtype=bool)
    target = np.array(target, dtype=bool)

    intersection = (output & target).sum()
    union = (output | target).sum()

    iou = (intersection + e) / (union + e)

    return iou


def metric_mse(output, target):
    return np.linalg.norm(output - target) / np.sqrt(target.size)


def metric_cd(output, target):
    distance_transform_target = scipy.ndimage.distance_transform_edt(1-target)
    return (distance_transform_target * output).sum() / target.size


def metric_dssim(output, target):
    return (1.0 - ssim(target, output, data_range=target.max() - target.min(), multichannel=True)) / 2.0
