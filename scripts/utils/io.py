import numpy as np
import cv2
import matplotlib.pyplot as plt
from scripts.utils.utils import nth_root, normalize
import scripts.utils.utils as utils
import torch
import math
from scipy import ndimage
import os


# expected volume shape: w x h x d x c
def volume_to_raw(volume, path, name, save_info=True):
    if volume.dtype == np.dtype(np.float32):
        volume = volume * 255.0
        volume = volume.astype(np.uint8)

    if save_info:
        f = open('{}/{}.raw.info'.format(path, name), 'w')
        f.write('Width {}\n'.format(volume.shape[0]))
        f.write('Height {}\n'.format(volume.shape[1]))
        f.write('Depth {}\n'.format(volume.shape[2]))
        f.write('WidthSize 1\n')
        f.write('HeightSize 1\n')
        f.write('DepthSize 1\n')
        f.write('BitsUsed {}\n'.format(8))
        f.write('ChannelsUsed {}\n'.format(volume.shape[3]))
        f.close()

    f = open('{}/{}.raw'.format(path, name), 'wb')
    f.write(bytearray(volume))
    f.close()


def read_image(image_path, resolution=None):
    # load image

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = utils.normalize(image)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    for channel in range(image.shape[2]):
        image[..., channel] = ndimage.filters.gaussian_filter(image[..., channel], 1, truncate=2.0)

    if resolution is not None:
        image = cv2.resize(image, (resolution, resolution), cv2.INTER_CUBIC).astype(np.float32)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    image = np.transpose(image, (2, 0, 1))

    return image


def read_volume(path, target_res=64):
    # read info file
    info_path = '{}.info'.format(path)
    if not os.path.isfile(info_path):
        raise FileNotFoundError(info_path)

    f = open(info_path, 'r')

    info = {}

    for line in f:
        key, value = line.strip().split(' ')

        info[key] = int(value)

    bytestream = np.fromfile(path, dtype=np.uint8)
    volume = np.reshape(bytestream, (info['Depth'], info['Height'], info['Width'], info['ChannelsUsed']))
    volume = np.flip(volume, axis=2).copy()
    volume = np.flip(volume, axis=1).copy()
    volume = np.flip(volume, axis=0).copy()
    volume = utils.convert_to_float(volume)

    for channel in range(info['ChannelsUsed']):
        volume[..., channel] = ndimage.filters.gaussian_filter(volume[..., channel], 1, truncate=2.0)

    volume = ndimage.interpolation.zoom(volume, (target_res / info['Depth'], target_res / info['Height'], target_res / info['Width'], 1), order=0)

    return volume


def read_vector_txt(path):
    vector = []
    with open('{}'.format(path), "r") as file:
        for line, content in enumerate(file):
            if line is not 0:
                vector.append(float(content))

    vector = np.array(vector)
    return vector


def volume_rotation_frames(volume, transform, renderer, param, direction='horizontal'):
    n_frames = 60

    bs = volume.shape[0]

    frames = torch.zeros((bs, n_frames, param.data.n_channel_out_3d, param.data.cube_len, param.data.cube_len))

    for frame_idx in range(0, n_frames):

        if direction == 'horizontal':
            theta = (frame_idx / n_frames) * np.pi * 2
            u = 0.0
        else:
            theta = 0.0
            u = ((frame_idx + 1) / (n_frames + 2)) * 2.0 - 1.0

        samples = torch.stack([torch.tensor(theta), torch.tensor(u)], dim=-1)
        vector = utils.polar_to_cartesian(samples.to(param.device))
        vector = vector.expand(bs, 3).to(param.device)

        volume_rotated = transform.rotate_by_vector(volume, vector)
        rendering = renderer.render(volume_rotated)
        frames[:, frame_idx, ...] = rendering

    return frames
