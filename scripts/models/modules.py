import torch.nn as nn
import torch
import copy
from torch.nn import functional as F
import math
from torch.nn import Parameter


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.stddev = lambda x: torch.sqrt(
            torch.mean((x - torch.mean(x, dim=0, keepdim=True)) ** 2, dim=0, keepdim=True) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        target_shape[1] = 1
        stddev = self.stddev(x)
        stddev = torch.mean(stddev, dim=1, keepdim=True)
        stddev = stddev.expand(*target_shape)
        output = torch.cat([x, stddev], 1)
        return output


class ResnetBasicBlock(nn.Module):
    def __init__(self, n_features_in, n_features_out, dim=2):
        super().__init__()
        # shortcut if features have different size
        self.shortcut = (n_features_in != n_features_out)
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.dim = dim

        self.conv_0, self.conv_0_bn = self._conv3x3(self.n_features_in, self.n_features_in)
        self.conv_1, self.conv_1_bn = self._conv3x3(self.n_features_in, self.n_features_out)

        if self.shortcut:
            self.conv_shortcut = self._conv1x1(self.n_features_in, self.n_features_out)

    def _conv3x3(self, n_features_in, n_features_out):
        if self.dim == 2:
            conv = nn.Conv2d(n_features_in, n_features_out, 3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm2d(n_features_out)

        elif self.dim == 3:
            conv = nn.Conv3d(n_features_in, n_features_out, 3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm3d(n_features_out)
        else:
            raise NotImplementedError

        return conv, bn

    def _conv1x1(self, n_features_in, n_features_out):
        if self.dim == 2:
            conv = nn.Conv2d(n_features_in, n_features_out, 1, stride=1, padding=0, bias=False)

        elif self.dim == 3:
            conv = nn.Conv3d(n_features_in, n_features_out, 1, stride=1, padding=0, bias=False)
        else:
            raise NotImplementedError

        return conv

    def forward(self, x):

        if self.shortcut:
            shortcut = self.conv_shortcut(x)
        else:
            shortcut = x

        x = F.leaky_relu(self.conv_0_bn(self.conv_0(x)), 0.2)
        x = F.leaky_relu(self.conv_1_bn(self.conv_1(x)), 0.2)

        x = x + shortcut

        return x
