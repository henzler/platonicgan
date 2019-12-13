import os
import matplotlib
import time, torch
from scripts.trainer import TrainerPlatonic, Trainer3D, TrainerPlatonic3D
from scripts.tests.tests import run_tests
from scripts.utils.config import load_config, make_dirs
import scripts.utils.utils as utils

# training setup
param = load_config()
dirs = make_dirs(param)

if param.mode == 'platonic':
    trainer = TrainerPlatonic(param, dirs)
elif param.mode == 'platonic_3D':
    trainer = TrainerPlatonic3D(param, dirs)
elif param.mode == '3D':
    trainer = Trainer3D(param, dirs)

if param.tests.activate:
    run_tests(trainer)

print('[Training] training start:')

for epoch in range(1, param.training.n_epochs + 1):
    start_time = time.time()

    for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(trainer.datal_loader_train):

        print("[iter {} name: {} mode: {} IF: {}]".format(trainer.logger.iteration, param.name, param.mode, param.renderer.type))

        image = image.to(param.device)
        volume = volume.to(param.device)

        if param.task == 'generation':
            z = utils.generate_z(param, param.training.z_size)
        else:
            z = trainer.encoder_train(image)

        trainer.generator_train(image, volume, z)
        trainer.discriminator_train(image, volume, z)

        trainer.logger.log_checkpoint(trainer.models, trainer.optimizers)
        for model in trainer.models:
            trainer.logger.log_gradients(model)

        trainer.logger.step()

print('Training finished!...')
