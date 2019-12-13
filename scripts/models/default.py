import torch
from torch import nn
from torch.nn import functional as F
from scripts.models.modules import ResnetBasicBlock


class Generator(nn.Module):
    def __init__(self, param):
        super(Generator, self).__init__()
        self.z_size = param.training.z_size

        self.n_features_min = param.models.generator.n_features_min
        self.n_channel = param.data.n_channel_out_3d
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len

        self.conv1 = nn.ConvTranspose3d(self.z_size, self.n_features_min * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(self.n_features_min * 8)
        self.conv2 = nn.ConvTranspose3d(self.n_features_min * 8, self.n_features_min * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.n_features_min * 4)
        self.conv3 = nn.ConvTranspose3d(self.n_features_min * 4, self.n_features_min * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.n_features_min * 2)
        self.conv4 = nn.ConvTranspose3d(self.n_features_min * 2, self.n_features_min * 1, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(self.n_features_min * 1)
        self.conv5 = nn.ConvTranspose3d(self.n_features_min * 1, self.n_channel, 4, 2, 1, bias=False)

    def forward(self, input):
        x = input.view(input.size(0), self.z_size, 1, 1, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
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
        self.param = param

        self.conv1, self.bn1 = self.get_conv(self.n_channel, self.n_features_min, 4, 2, 1)
        self.conv2, self.bn2 = self.get_conv(self.n_features_min, self.n_features_min * 2, 4, 2, 1)
        self.conv3, self.bn3 = self.get_conv(self.n_features_min * 2, self.n_features_min * 4, 4, 2, 1)
        self.conv4, self.bn4 = self.get_conv(self.n_features_min * 4, self.n_features_min * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(self.n_features_min * 8, 1, 4, 1, 0)

    def get_conv(self, n_features_in, n_features_out, kernel, stride, padding):
        return nn.Conv2d(n_features_in, n_features_out, kernel, stride, padding), nn.BatchNorm2d(n_features_out)

    def forward(self, input):
        x_f1 = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x_f1)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x_f2 = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x_f2)

        return x, [x_f1, x_f2]


class Discriminator3d(nn.Module):
    def __init__(self, param):
        super(Discriminator3d, self).__init__()
        self.z_size = param.training.z_size
        self.n_features_min = param.models.discriminator.n_features_min
        self.n_channel = param.data.n_channel_out_3d
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len
        self.param = param

        self.conv1, self.bn1 = self.get_conv(self.n_channel, self.n_features_min, 4, 2, 1)
        self.conv2, self.bn2 = self.get_conv(self.n_features_min, self.n_features_min * 2, 4, 2, 1)
        self.conv3, self.bn3 = self.get_conv(self.n_features_min * 2, self.n_features_min * 4, 4, 2, 1)
        self.conv4, self.bn4 = self.get_conv(self.n_features_min * 4, self.n_features_min * 8, 4, 2, 1)
        self.conv5 = nn.Conv3d(self.n_features_min * 8, 1, 4, 1, 0)

    def get_conv(self, n_features_in, n_features_out, kernel, stride, padding):
        return nn.Conv3d(n_features_in, n_features_out, kernel, stride, padding), nn.BatchNorm3d(n_features_out)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        x_f1 = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x_f1)), 0.2)
        x_f2 = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x_f2)

        return x, [x_f1, x_f2]


class Encoder(nn.Module):
    def __init__(self, param):
        super(Encoder, self).__init__()
        self.z_size = param.training.z_size
        self.n_features_min = param.models.discriminator.n_features_min
        self.n_channel = param.data.n_channel_in
        self.batch_size = param.training.batch_size
        self.cube_len = param.data.cube_len

        self.conv1 = nn.Conv2d(self.n_channel, self.n_features_min, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.n_features_min)

        self.conv2 = nn.Conv2d(self.n_features_min, self.n_features_min * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.n_features_min * 2)

        self.conv3 = nn.Conv2d(self.n_features_min * 2, self.n_features_min * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.n_features_min * 4)

        self.conv4 = nn.Conv2d(self.n_features_min * 4, self.n_features_min * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.n_features_min * 8)

        self.conv5 = nn.Conv2d(self.n_features_min * 8, self.n_features_min * 16, 4, 1, 0)
        self.bn5 = nn.BatchNorm2d(self.n_features_min * 16)

        self.fc = nn.Linear(self.n_features_min * 16, self.z_size)

    def forward(self, input):
        batch_size = input.size(0)
        layer1 = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        layer2 = F.leaky_relu(self.bn2(self.conv2(layer1)), 0.2)
        layer3 = F.leaky_relu(self.bn3(self.conv3(layer2)), 0.2)
        layer4 = F.leaky_relu(self.bn4(self.conv4(layer3)), 0.2)
        layer5 = F.leaky_relu(self.bn5(self.conv5(layer4)), 0.2)
        layer6 = layer5.view(batch_size, self.n_features_min * 16)
        layer6 = self.fc(layer6)
        return layer6
