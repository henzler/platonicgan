import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from scripts.trainer.trainer import Trainer
import torch.nn as nn
import torchvision.models as models


class TrainerPlatonic(Trainer):
    def __init__(self, param, dirs, test=False, init=True):
        print("[INFO] setup TrainerPlatonic")

        Trainer.__init__(self, param, dirs, test=test, init=init)

        self.d_optimizer = None

        if init:
            self.discriminator = self.discriminator_2d
            if not test:
                self.d_optimizer = self.d_optimizer_2d

    def process_views(self, volume, image_real, target):

        views = []
        losses = []
        d_fakes = []
        gradient_penalties = []

        n_views = self.param.training.n_views

        if self.param.task == 'reconstruction':
            n_views += 1

        for idx in range(n_views):
            if idx == 0 and self.param.task == 'reconstruction':
                view = self.renderer.render(volume)
            else:
                view = self.renderer.render(self.transform.rotate_random(volume))
            d_fake, _ = self.discriminator(view)
            loss = self._compute_adversarial_loss(d_fake, target, self.param.training.adversarial_term_lambda_2d)
            gp = self.gan_loss.gradient_penalty(image_real, view, self.discriminator)
            self.d_optimizer.zero_grad()

            self.logger.log_images('{}_{}'.format('view_output', idx), view)

            views.append(view)
            losses.append(loss)
            gradient_penalties.append(gp)
            d_fakes.append(d_fake)

        return views, d_fakes, losses, gradient_penalties

    def generator_train(self, image, volume, z):
        self.generator.train()
        self.g_optimizer.zero_grad()

        data_loss = torch.tensor(0.0).to(self.param.device)

        fake_volume = self.generator(z)

        if self.param.task == 'reconstruction':
            fake_front = self.renderer.render(fake_volume)
            data_loss += self._compute_data_term_loss(fake_front, image, self.param.training.data_term_lambda_2d)

        view, _, losses, _ = self.process_views(fake_volume, image, 1)

        g_loss = torch.mean(torch.stack(losses)) + data_loss

        g_loss.backward()
        self.g_optimizer.step()

        ### log
        print("  Generator loss 2d: {:2.4}".format(g_loss))
        self.logger.log_scalar('g_2d_loss', g_loss.item())
        self.logger.log_scalar('g_2d_rec_loss', data_loss.item())
        self.logger.log_volumes('volume', fake_volume)
        self.logger.log_images('image_input', image)

    def discriminator_train(self, image, volume, z):
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        d_real, _ = self.discriminator(image)
        d_real_loss = self._compute_adversarial_loss(d_real, 1, self.param.training.adversarial_term_lambda_2d)

        with torch.no_grad():
            fake_volume = self.generator(z)

        view, d_fakes, losses, gradient_penalties = self.process_views(fake_volume, image, 0)

        gradient_penalty = torch.mean(torch.stack(gradient_penalties))
        d_fake_loss = torch.mean(torch.stack(losses)) + gradient_penalty
        d_loss = d_real_loss + d_fake_loss

        if self.param.training.loss == 'vanilla':
            d_real = torch.sigmoid(d_real)
            d_fakes = list(map(torch.sigmoid, d_fakes))

            d_real_accuracy = torch.mean(d_real)
            d_fake_accuracy = 1.0 - torch.mean(torch.stack(d_fakes))
            d_accuracy = ((d_real_accuracy + d_fake_accuracy) / 2.0).item()
        else:
            d_accuracy = 0.0

        # only update discriminator if accuracy <= d_thresh
        if d_accuracy <= self.param.training.d_thresh or self.param.training.loss == 'wgangp':
            d_loss.backward()
            self.d_optimizer.step()
            print("  *Discriminator 2d update*")

        ### log
        print("  Discriminator2d loss: {:2.4}".format(d_loss))
        self.logger.log_scalar('d_2d_loss', d_loss.item())
        self.logger.log_scalar('d_2d_real', torch.mean(d_real).item())
        self.logger.log_scalar('d_2d_fake', torch.mean(torch.stack(d_fakes)).item())
        self.logger.log_scalar('d_2d_real_loss', d_real_loss.item())
        self.logger.log_scalar('d_2d_fake_loss', d_fake_loss.item())
        self.logger.log_scalar('d_2d_accuracy', d_accuracy)
        self.logger.log_scalar('d_2d_gradient_penalty', gradient_penalty.item())
