import torch
from scripts.trainer.trainer import Trainer


class Trainer3D(Trainer):
    def __init__(self, param, dirs, test=False, init=True):

        Trainer.__init__(self, param, dirs, test=test, init=init)
        print("[INFO] setup Trainer3D")

        self.d_optimizer = None
        if init:
            self.discriminator = self.discriminator_3d
            self.d_optimizer = self.d_optimizer_3d

    def generator_train(self, image, volume, z):

        self.g_optimizer.zero_grad()

        fake_volume = self.generator(z)

        self.logger.log_volumes('volume', fake_volume)

        d_fake, _ = self.discriminator(fake_volume)
        adversarial_loss = self._compute_adversarial_loss(d_fake, 1, self.param.training.adversarial_term_lambda_3d)
        data_loss = self._compute_data_term_loss(fake_volume, volume, self.param.training.data_term_lambda_3d)

        g_loss = adversarial_loss + data_loss
        g_loss.backward()
        self.g_optimizer.step()

        print("  Generator loss 3d: {:2.4}".format(g_loss))
        self.logger.log_scalar('g_data_loss_3d', data_loss.item())
        self.logger.log_scalar('g_adversrarial_loss_3d', adversarial_loss.item())

    def discriminator_train(self, image, volume, z):

        self.g_optimizer.zero_grad()

        with torch.no_grad():
            fake_volume = self.generator(z)

        d_fake, _ = self.discriminator(fake_volume)
        d_real, _ = self.discriminator(volume)

        d_fake_loss = self._compute_adversarial_loss(d_fake, 0, self.param.training.adversarial_term_lambda_3d)
        d_real_loss = self._compute_adversarial_loss(d_real, 1, self.param.training.adversarial_term_lambda_3d)

        gp = self.gan_loss.gradient_penalty(volume, fake_volume, self.discriminator)

        d_loss = d_fake_loss + d_real_loss + gp

        d_real_accuracy = torch.mean(d_real)
        d_fake_accuracy = 1.0 - torch.mean(d_fake)
        d_accuracy = ((d_real_accuracy + d_fake_accuracy) / 2.0).item()

        # only update discriminator if accuracy <= d_thresh
        if d_accuracy <= self.param.training.d_thresh or self.param.training.loss == 'wgangp':

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
            print("  *Discriminator 3d update*")

        self.logger.log_scalar('d_3d_loss', d_loss.item())
        self.logger.log_scalar('d_3d_fake', d_fake[0].item())
        self.logger.log_scalar('d_3d_real', d_real[0].item())

        print("  Discriminator loss 3d: {:2.4}".format(d_loss))
        self.logger.log_scalar('d_3d_loss', d_loss.item())
