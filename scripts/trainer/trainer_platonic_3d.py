import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from scripts.trainer.trainer import Trainer
from scripts.trainer import TrainerPlatonic, Trainer3D


class TrainerPlatonic3D(Trainer):
    def __init__(self, param, dirs, test=False):
        print("[INFO] setup TrainerPlatonic3D")

        Trainer.__init__(self, param, dirs, test=test)

        self.trainer_platonic = TrainerPlatonic(param, dirs, test=test, init=False)
        self.trainer_3d = Trainer3D(param, dirs, test=test, init=False)

        self.trainer_platonic.encoder = self.encoder
        self.trainer_platonic.generator = self.generator
        self.trainer_platonic.discriminator = self.discriminator_2d
        self.trainer_platonic.renderer = self.renderer
        self.trainer_platonic.transform = self.transform

        self.trainer_3d.encoder = self.encoder
        self.trainer_3d.generator = self.generator
        self.trainer_3d.discriminator = self.discriminator_3d
        self.trainer_3d.renderer = self.renderer
        self.trainer_3d.transform = self.transform

        if not test:
            self.trainer_platonic.g_optimizer = self.g_optimizer
            self.trainer_platonic.d_optimizer = self.d_optimizer_2d
            self.trainer_platonic.logger = self.logger
            self.trainer_platonic.gan_loss = self.gan_loss
            self.trainer_platonic.criterion_data_term = self.criterion_data_term

            self.trainer_3d.g_optimizer = self.g_optimizer
            self.trainer_3d.d_optimizer = self.d_optimizer_3d
            self.trainer_3d.logger = self.logger
            self.trainer_3d.gan_loss = self.gan_loss
            self.trainer_3d.criterion_data_term = self.criterion_data_term
            self.dirs = dirs

    def generator_train(self, image, volume, z):
        self.trainer_3d.generator_train(image, volume, z)

    def discriminator_train(self, image, volume, z):
        self.trainer_platonic.discriminator_train(image, volume, z)
        self.trainer_3d.discriminator_train(image, volume, z)
