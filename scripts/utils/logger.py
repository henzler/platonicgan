import matplotlib.pyplot as plt
import scripts.utils.io as dh
import scripts.utils.utils as utils
import numpy as np
import torch
import imageio
import os
import yaml
import cv2
import math
import tensorflow as tf
import tensorboardX
from scripts.utils.torchsummary import summary


class Logger():
    def __init__(self, dirs, param, renderer, transform, models, optimizers):

        print("[INFO] setup logger")

        self.iteration = 0
        self.logging_scalars = False
        self.logging_files = False
        self.param = param
        self.num_examples_log = param.logger.num_examples
        self.log_stats = dict()
        self.path_stats = dirs.stats
        self.writer = tensorboardX.SummaryWriter(self.path_stats)
        self.transform = transform
        self.models = models
        self.optimizers = optimizers
        self.renderer = renderer

        self.load_checkpoint()
        self.write_models()
        self.write_config()

    def write_models(self):
        # generate model file
        model_filename = open('{}/model.txt'.format(self.path_stats), "a")
        model_filename.seek(0)
        model_filename.truncate()

        summary(self.models[1], input_size=(1, self.param.training.z_size), device=self.param.device, file=model_filename)

        if self.models[0] is not None:
            summary(self.models[0],
                    input_size=(self.param.data.n_channel_in, self.param.data.cube_len, self.param.data.cube_len),
                    device=self.param.device,
                    file=model_filename)

        if self.models[2] is not None:
            summary(self.models[2],
                    input_size=(self.param.data.n_channel_out_2d, self.param.data.cube_len, self.param.data.cube_len),
                    device=self.param.device, file=model_filename)

        if self.models[3] is not None:
            summary(self.models[3],
                    input_size=(self.param.data.n_channel_out_3d, self.param.data.cube_len, self.param.data.cube_len, self.param.data.cube_len),
                    device=self.param.device, file=model_filename)

        model_filename.close()

    def write_config(self):

        # generate config file
        config_file = open('{}/config.txt'.format(self.path_stats), "a")
        if not self.param.training.resume:
            config_file.seek(0)
            config_file.truncate()
        yaml.dump(self.param, config_file, default_flow_style=False)
        config_file.write('job_id: {}\n'.format(self.param.job_id))
        config_file.flush()
        config_file.close()

    def step(self):

        self.iteration += 1

        if self.iteration % self.param.logger.log_scalars_every == 0:
            self.logging_scalars = True
        else:
            self.logging_scalars = False

        if self.iteration % self.param.logger.log_files_every == 0:
            self.logging_files = True
        else:
            self.logging_files = False

    def log_scalar(self, tag, value):
        if self.logging_scalars:
            self.writer.add_scalar(tag, value, self.iteration)

    def log_gradients(self, model):

        if self.logging_scalars and self.param.logger.log_gradients and model is not None:
            for name, param in model.named_parameters():
                self.writer.add_histogram(model.__class__.__name__ + '_' + name, param.clone().cpu().data.numpy(), self.iteration)

    def load_checkpoint(self):

        try:
            checkpoint = torch.load('{}/checkpoint.pkl'.format(self.path_stats))

            self.iteration = checkpoint['iteration'] + 1

            for idx, model in enumerate(self.models):
                if model is not None:
                    model.load_state_dict(checkpoint['model_{}'.format(str(idx))])

            for idx, optimizer in enumerate(self.optimizers):
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint['optimizer_{}'.format(str(idx))])

            print('[Training] checkpoint loaded:')

        except FileNotFoundError:
            print('[Info] No checkpoint file available')
            pass

    def log_checkpoint(self, models, optimizers):

        if self.logging_files:

            checkpoint = {
                'iteration': self.iteration,
            }

            for idx, model in enumerate(models):
                if model is not None:
                    checkpoint['model_{}'.format(idx)] = model.state_dict()

            for idx, optimizer in enumerate(optimizers):
                if optimizer is not None:
                    checkpoint['optimizer_{}'.format(idx)] = optimizer.state_dict()

            torch.save(checkpoint, '{}/checkpoint.pkl'.format(self.path_stats))

    def log_images(self, tag, images):
        if self.logging_files:

            images = images.clone()
            images = images.detach()

            with torch.no_grad():

                if self.param.renderer.type == 'emission_absorption':

                    images = torch.clamp(images, min=0.0, max=1.0)
                    images_rgb = images[:, :3, ...]
                    images_a = images[:, [3], ...]
                    images_rgb = images_rgb * images_a
                    images_a = torch.cat([images_a, images_a, images_a], dim=1)

                    self.writer.add_images('{}_rgb'.format(tag), images_rgb, self.iteration)
                    self.writer.add_images('{}_a'.format(tag), images_a, self.iteration)
                else:
                    self.writer.add_images('{}_a'.format(tag), torch.cat([images, images, images], dim=1), self.iteration)

    def log_volumes(self, tag, volumes):
        if self.logging_files:
            volumes = volumes.clone()
            volumes = volumes.detach()

            with torch.no_grad():
                spinx = dh.volume_rotation_frames(volumes, self.transform, self.renderer, self.param, direction='horizontal')
                spiny = dh.volume_rotation_frames(volumes, self.transform, self.renderer, self.param, direction='vertical')

                def log_video(tag, video):
                    if self.param.renderer.type == 'emission_absorption':
                        videox_rgb = video[:, :, :3, ...]
                        videox_a = video[:, :, [3], ...]
                        self.writer.add_video('{}_x_rgb'.format(tag), videox_rgb, self.iteration)
                        self.writer.add_video('{}_x_a'.format(tag), videox_a, self.iteration)
                    else:
                        self.writer.add_video('{}_x_a'.format(tag), video, self.iteration)

                log_video('{}_horizontal'.format(tag), spinx)
                log_video('{}_vertical'.format(tag), spiny)
