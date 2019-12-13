import torch
from torch import nn
from torch.nn import functional as F
from scripts.models.modules import ResnetBasicBlock
from scripts.models.modules import MinibatchStddev


class Generator(nn.Module):
    def __init__(self, param):
        super(Generator, self).__init__()
        self.z_size = param.training.z_size
        self.n_features_min = param.models.generator.n_features_min
        self.n_channel = param.data.n_channel_out_3d
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len

        self.conv1 = nn.Conv3d(self.z_size, self.n_features_min * 16, 4, 1, 3)
        self.conv1_bn = nn.BatchNorm3d(self.n_features_min * 16)

        self.resnet_0_0 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 16, dim=3)
        self.resnet_0_1 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 16, dim=3)

        self.resnet_1_0 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 8, dim=3)
        self.resnet_1_1 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 8, dim=3)

        self.resnet_2_0 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 4, dim=3)
        self.resnet_2_1 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 4, dim=3)

        self.resnet_3_0 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 2, dim=3)
        self.resnet_3_1 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min * 2, dim=3)

        self.resnet_4_0 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min, dim=3)
        self.resnet_4_1 = ResnetBasicBlock(self.n_features_min, self.n_features_min, dim=3)

        self.conv = nn.Conv3d(self.n_features_min, self.n_channel, 1, 1, 0)

    def forward(self, input):
        x = input.view(input.size(0), self.z_size, 1, 1, 1)

        x = F.relu(self.conv1_bn(self.conv1(x)))

        x = self.resnet_0_0(x)
        x = self.resnet_0_1(x)

        x = F.upsample(x, scale_factor=2)
        x = self.resnet_1_0(x)
        x = self.resnet_1_1(x)

        x = F.upsample(x, scale_factor=2)
        x = self.resnet_2_0(x)
        x = self.resnet_2_1(x)

        x = F.upsample(x, scale_factor=2)
        x = self.resnet_3_0(x)
        x = self.resnet_3_1(x)

        x = F.upsample(x, scale_factor=2)
        x = self.resnet_4_0(x)
        x = self.resnet_4_1(x)

        x = self.conv(x)
        x = torch.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        self.z_size = param.training.z_size
        self.n_features_min = param.models.discriminator.n_features_min
        self.n_channel = param.data.n_channel_out_2d
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len

        self.conv = nn.Conv2d(self.n_channel, self.n_features_min, 1, 1)

        self.resnet_0_0 = ResnetBasicBlock(self.n_features_min, self.n_features_min)
        self.resnet_0_1 = ResnetBasicBlock(self.n_features_min, self.n_features_min * 2)

        self.resnet_1_0 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min * 2)
        self.resnet_1_1 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min * 4)

        self.resnet_2_0 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 4)
        self.resnet_2_1 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 8)

        self.resnet_3_0 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 8)
        self.resnet_3_1 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 16)

        self.resnet_4_0 = ResnetBasicBlock(self.n_features_min * 16 + 1, self.n_features_min * 16)
        self.resnet_4_1 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 16)

        self.fc = nn.Linear(self.n_features_min * 16 * 16, 1)

        self.minibatch_stddev = MinibatchStddev()

    def forward(self, input):
        x = self.conv(input)

        x = self.resnet_0_0(x)
        x = self.resnet_0_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_1_0(x)
        x = self.resnet_1_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_2_0(x)
        x = self.resnet_2_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_3_0(x)
        x = self.resnet_3_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.minibatch_stddev(x)

        x = self.resnet_4_0(x)
        x = self.resnet_4_1(x)

        x = x.view(x.size(0), self.n_features_min * 16 * 16)
        x = self.fc(x)

        return x, []


class Encoder(nn.Module):
    def __init__(self, param):
        super(Encoder, self).__init__()
        self.z_size = param.training.z_size
        self.n_features_min = param.models.encoder.n_features_min
        self.n_channel = param.data.n_channel_in
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len
        self.param = param

        self.conv = nn.Conv2d(self.n_channel, self.n_features_min, 1, 1)

        self.resnet_0_0 = ResnetBasicBlock(self.n_features_min, self.n_features_min)
        self.resnet_0_1 = ResnetBasicBlock(self.n_features_min, self.n_features_min * 2)

        self.resnet_1_0 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min * 2)
        self.resnet_1_1 = ResnetBasicBlock(self.n_features_min * 2, self.n_features_min * 4)

        self.resnet_2_0 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 4)
        self.resnet_2_1 = ResnetBasicBlock(self.n_features_min * 4, self.n_features_min * 8)

        self.resnet_3_0 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 8)
        self.resnet_3_1 = ResnetBasicBlock(self.n_features_min * 8, self.n_features_min * 16)

        self.resnet_4_0 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 16)
        self.resnet_4_1 = ResnetBasicBlock(self.n_features_min * 16, self.n_features_min * 16)

        self.fc = nn.Linear(self.n_features_min * 16 * 16, self.z_size)

    def forward(self, input):
        x = self.conv(input)

        x = self.resnet_0_0(x)
        x = self.resnet_0_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_1_0(x)
        x = self.resnet_1_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_2_0(x)
        x = self.resnet_2_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_3_0(x)
        x = self.resnet_3_1(x)
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.resnet_4_0(x)
        x = self.resnet_4_1(x)

        x = x.view(x.size(0), self.n_features_min * 16 * 16)

        x = self.fc(x)

        return x
