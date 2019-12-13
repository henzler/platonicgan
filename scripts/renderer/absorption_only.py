import torch


class AbsorptionOnlyRenderer():
    def __init__(self, param):
        self.param = param
        self.absorption_factor = param.renderer.absorption_factor

    def render(self, volume, axis=2):
        volume = (volume * self.absorption_factor) / self.param.data.cube_len
        volume = 1.0 - volume

        volume = torch.clamp(volume, 0.00001, 1.0)

        # start log space for better numerical stability
        volume = torch.log(volume)
        image = torch.sum(volume, dim=axis)
        image = torch.exp(image)
        # exit log space

        image = 1.0 - image

        return image
