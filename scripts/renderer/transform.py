import torch
import numpy as np
import scipy.ndimage
import math
import scripts.utils.utils as utils


class Transform():
    def __init__(self, param):

        print("[INFO] setup transform")

        self.param = param

    def rotate(self, volume, rotation_matrix):

        batch_size = volume.shape[0]
        size = volume.shape[2]

        indices = utils.get_graphics_grid_coords_3d(size, size, size, coord_dim=0)
        indices = indices.expand(batch_size, 3, size, size, size)
        indices = indices.to(self.param.device)

        indices_rotated = torch.bmm(rotation_matrix, indices.view(batch_size, 3, -1)).view(batch_size, 3, size, size, size)

        return self.resample(volume, indices_rotated)

    def rotate_random(self, volume):

        batch_size = volume.shape[0]
        rotation_matrix = self.get_rotation_matrix_random(batch_size)

        return self.rotate(volume, rotation_matrix)

    def rotate_by_vector(self, volume, vector):

        rotation_matrix = self.get_view_matrix_by_vector(vector)

        return self.rotate(volume, rotation_matrix)

    def get_vector_random(self, batch_size):

        theta = torch.empty((batch_size), dtype=torch.float).uniform_(0, 2 * np.pi).to(self.param.device)
        u = torch.empty((batch_size), dtype=torch.float).uniform_(-1.0, 1.0).to(self.param.device)

        x = torch.sqrt(1 - u * u) * torch.cos(theta)
        y = torch.sqrt(1 - u * u) * torch.sin(theta)
        z = u

        vector = torch.stack([x, y, z], dim=1)

        return vector

    def get_rotation_matrix_random(self, batch_size):

        vector = self.get_vector_random(batch_size)
        random_view_matrix = self.get_view_matrix_by_vector(vector)
        return random_view_matrix

    def get_view_matrix_by_vector(self, eye, up_vector=None, target=None):

        bs = eye.size(0)

        if up_vector is None:
            up_vector = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0]).expand(bs, 3), dim=1).to(self.param.device)

        if target is None:
            target = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 0.0]).expand(bs, 3), dim=1).to(self.param.device)

        z_axis = torch.nn.functional.normalize(eye - target, dim=1).expand(bs, 3)
        x_axis = torch.nn.functional.normalize(torch.cross(up_vector, z_axis), dim=1)
        y_axis = torch.cross(z_axis, x_axis, dim=1)
        view_matrix = torch.stack([x_axis, y_axis, z_axis], dim=2)

        return view_matrix.to(self.param.device)

    def resample(self, volume, indices_rotated):

        if volume.is_cuda:
            indices_rotated = indices_rotated.to('cuda')

        indices_rotated = indices_rotated.permute(0, 2, 3, 4, 1)

        # transform coordinate system
        # flip y and z
        # grid sample expects y- to be up and z- to be front
        indices_rotated[..., 1] = -indices_rotated[..., 1]
        indices_rotated[..., 2] = -indices_rotated[..., 2]
        volume = torch.nn.functional.grid_sample(volume, indices_rotated, mode='bilinear')

        return volume
