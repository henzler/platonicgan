import torch


class EmissionAbsorptionRenderer():
    def __init__(self, param):
        self.param = param

        self.density_factor = 100
        self.sample_size_z = 1024

        self.counter = 0

        # pre-compute bilinearly sampled coordinates
        steps_z = torch.linspace(-1.0, 1.0, self.sample_size_z)
        steps_xy = torch.linspace(-1.0, 1.0, self.param.data.cube_len)
        self.sample_coords = self.get_grid_coords_3d(steps_z, steps_xy, steps_xy).to(self.param.device)

    def get_grid_coords_3d(self, z, y, x, coord_dim=-1):
        z, y, x = torch.meshgrid(z, y, x)
        coords = torch.stack([x, y, z], dim=coord_dim)
        return coords

    def render(self, volume, axis=2):
        padding = 1
        volume = torch.nn.functional.pad(volume, (padding, padding, padding, padding, padding, padding))

        density = volume[:, [3]]
        signal = volume[:, :3]

        bs = density.shape[0]
        sample_coords = self.sample_coords.expand(bs, self.sample_size_z, self.param.data.cube_len, self.param.data.cube_len, 3)

        density = density * self.density_factor
        density = density / self.sample_size_z
        density = torch.nn.functional.grid_sample(density, sample_coords)
        transmission = torch.cumprod(1.0 - density, dim=axis)

        weight = density * transmission
        weight_sum = torch.sum(weight, dim=axis)

        signal = torch.nn.functional.grid_sample(signal, sample_coords)

        rendering = torch.sum(weight * signal, dim=axis)
        rendering = rendering / (weight_sum + 1e-8)

        alpha = 1.0 - torch.prod(1 - density, dim=axis)

        rendering = rendering * alpha

        rendering = torch.cat([rendering, alpha], dim=1)

        return rendering
