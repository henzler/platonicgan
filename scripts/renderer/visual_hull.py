import torch
from scripts.renderer.absorption_only import AbsorptionOnlyRenderer


class VisualHullRenderer():
    def __init__(self, param):
        self.param = param

    def render(self, volume, axis=2):
        image = torch.sum(volume, dim=axis)
        image = torch.ones_like(image) - torch.exp(-image)

        return image
