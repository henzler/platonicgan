import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
import scripts.utils.logger as log
from torch import optim
from scripts.data import dataset_dict
from scripts.renderer import renderer_dict
from scripts.models import generator_dict, discriminator_dict, encoder_dict
import scripts.renderer.transform as dt


class Trainer(object):
    def __init__(self, param, dirs, test=False, init=True):

        self.param = param
        self.iteration = 0

        if init:
            self.datal_loader_train, self.dataset_train = self.setup_dataset('train')
            self.data_loader_val, self.dataset_val = self.setup_dataset('val')
            self.data_loader_test, self.dataset_test = self.setup_dataset('test')

            self.models = self.setup_models()
            self.encoder = self.models[0]
            self.generator = self.models[1]
            self.discriminator_2d = self.models[2]
            self.discriminator_3d = self.models[3]

            self.renderer = self.setup_renderer()
            self.transform = dt.Transform(param)

            if not test:
                self.optimizers = self.setup_optimizers()
                self.g_optimizer = self.optimizers[0]
                self.d_optimizer_2d = self.optimizers[1]
                self.d_optimizer_3d = self.optimizers[2]
                self.logger = log.Logger(dirs, param, self.renderer, self.transform, self.models, self.optimizers)
                self.gan_loss = GANLoss(param, param.training.loss)
                self.criterion_data_term = torch.nn.MSELoss()

                self.dirs = dirs

    def encoder_train(self, x):
        self.encoder.train()
        z = self.encoder(x)

        return z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def _compute_adversarial_loss(self, output, target, weight):
        return self.gan_loss(output, target) * weight

    def _compute_data_term_loss(self, output, target, weight):
        return self.criterion_data_term(output, target) * weight

    def setup_models(self):
        print("[INFO] setup models")

        # encoder
        encoder = None
        if self.param.task == 'reconstruction':
            encoder_type = encoder_dict[self.param.models.encoder.name]
            encoder = encoder_type(self.param)
            encoder = encoder.to(self.param.device)

        # generator
        generator_type = generator_dict[self.param.models.generator.name]
        generator = generator_type(self.param)
        generator = generator.to(self.param.device)

        # discriminator
        discriminator_2d = None
        discriminator_3d = None

        if self.param.mode == '3D' or self.param.mode == 'platonic_3D':
            discriminator_type = discriminator_dict['{}_3d'.format(self.param.models.discriminator.name)]
            discriminator_3d = discriminator_type(self.param)
            discriminator_3d = discriminator_3d.to(self.param.device)

        if self.param.mode == 'platonic' or self.param.mode == 'platonic_3D':
            discriminator_type = discriminator_dict[self.param.models.discriminator.name]
            discriminator_2d = discriminator_type(self.param)
            discriminator_2d = discriminator_2d.to(self.param.device)

        return [encoder, generator, discriminator_2d, discriminator_3d]

    # type =  {train, test, val}
    def setup_dataset(self, type):
        print("[INFO] setup dataset {}".format(type))

        dataset_type = dataset_dict[self.param.data.name]
        dataset = dataset_type(self.param, type)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.param.training.batch_size, shuffle=True, num_workers=self.param.n_workers, drop_last=True)

        dataset_len = dataset.__len__()
        if dataset_len == 0 or dataset_len < self.param.n_workers or dataset_len < self.param.training.batch_size:
            print("Dataset ({}) does not contain enough samples".format(self.param.data.path_dir))
            exit(1)

        return dataloader, dataset

    def setup_renderer(self):
        print("[INFO] generate renderer")
        renderer_type = renderer_dict[self.param.renderer.type]
        renderer = renderer_type(self.param)

        return renderer

    def setup_optimizers(self):
        print("[INFO] generate optimizers")

        optimizer_g = self.param.training.optimizer_g
        optimizer_d = self.param.training.optimizer_d
        lr_g = self.param.training.lr_g
        lr_d = self.param.training.lr_d

        if self.param.task == 'reconstruction':
            g_params = list(self.models[0].parameters()) + list(self.models[1].parameters())
        else:
            g_params = list(self.models[1].parameters())

        d_optimizer_2d = None
        d_optimizer_3d = None

        if self.param.mode == 'platonic_3D' or self.param.mode == '3D':
            d_params_3d = filter(lambda p: p.requires_grad, self.models[3].parameters())
            if optimizer_d == 'rmsprop':
                d_optimizer_3d = optim.RMSprop(d_params_3d, lr=lr_d, alpha=0.99, eps=1e-8)
            elif optimizer_d == 'adam':
                d_optimizer_3d = optim.Adam(d_params_3d, lr=lr_d, betas=(0.5, 0.9))
            elif optimizer_d == 'sgd':
                d_optimizer_3d = optim.SGD(d_params_3d, lr=lr_d)
            else:
                raise NotImplementedError

        if self.param.mode == 'platonic' or self.param.mode == 'platonic_3D':

            d_params_2d = filter(lambda p: p.requires_grad, self.models[2].parameters())
            if optimizer_d == 'rmsprop':
                d_optimizer_2d = optim.RMSprop(d_params_2d, lr=lr_d, alpha=0.99, eps=1e-8)
            elif optimizer_d == 'adam':
                d_optimizer_2d = optim.Adam(d_params_2d, lr=lr_d, betas=(0.5, 0.9))
            elif optimizer_d == 'sgd':
                d_optimizer_2d = optim.SGD(d_params_2d, lr=lr_d)
            else:
                raise NotImplementedError

        if optimizer_g == 'rmsprop':
            g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        elif optimizer_g == 'adam':
            g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.9))
        elif optimizer_g == 'sgd':
            g_optimizer = optim.SGD(g_params, lr=lr_g)
        else:
            raise NotImplementedError

        return [g_optimizer, d_optimizer_2d, d_optimizer_3d]


class GANLoss(object):

    def __init__(self, param, loss_type):

        super(GANLoss, self).__init__()
        self.param = param
        self.loss_type = loss_type

        if self.loss_type == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.loss_type in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('GAN type {} not implemented'.format(self.loss_type))

    def __call__(self, prediction, target):
        if self.loss_type in ['vanilla']:
            targets = prediction.new_full(size=prediction.size(), fill_value=target)
            loss = self.loss(prediction, targets)
        elif self.loss_type == 'wgangp':
            if target == 1.0:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss

    def gradient_penalty(self, real, fake, discriminator):

        if self.loss_type == 'wgangp':

            real_data = real.data
            fake_data = fake.data

            # Random weight term for interpolation between real and fake samples
            ndims = len(real_data.shape)
            alpha = torch.rand((real_data.size(0), 1)).to(self.param.device)
            if ndims == 5:
                alpha = alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                alpha = alpha.unsqueeze(1).unsqueeze(1)

            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

            d_interpolates, _ = discriminator(interpolates)
            fake = torch.ones(d_interpolates.size(), dtype=torch.float, requires_grad=True).to(self.param.device)

            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = (((gradients + 1e-8).norm(2, dim=1) - 1.0) ** 2).mean()

            gradient_penalty = 10 * gradient_penalty
        else:
            gradient_penalty = torch.tensor(0.0)

        return gradient_penalty
