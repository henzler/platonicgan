import cv2
import torch
import numpy as np
import os
import scripts.utils.io as dh
import scripts.utils.utils as utils
from scipy import ndimage


def run_test_render(trainer):
    renderer = trainer.renderer

    volume = dh.read_volume(os.path.join(os.getcwd(), 'scripts/tests/input_data/object.raw'))
    vector = dh.read_vector_txt(os.path.join(os.getcwd(), 'scripts/tests/input_data/vector.txt'))

    volume = utils.numpy_to_pytorch(volume)
    volume = volume.to(trainer.param.device)

    if trainer.param.data.n_channel_out_3d == 1:
        volume = volume[[3]]

    volume = volume[np.newaxis, :trainer.param.data.n_channel_out_3d, :, :, :]
    vector = torch.from_numpy(vector[np.newaxis, ...]).float().to(trainer.param.device)

    volume_rotated = trainer.transform.rotate_by_vector(volume, vector)
    image = renderer.render(volume_rotated)
    image = utils.pytorch_to_numpy(image)
    image = utils.convert_to_int(image)[0]
    cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_vector.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for i in range(0, 5):
        volume_rotated = trainer.transform.rotate_random(volume)
        image = renderer.render(volume_rotated)
        image = utils.pytorch_to_numpy(image)
        image = utils.convert_to_int(image)[0]
        cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_train{}.png'.format(i)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    volume_rotated = trainer.transform.rotate_by_vector(volume, vector=torch.tensor([[0.0, 0.0, 1.0]], device=trainer.param.device))
    image = renderer.render(volume_rotated)
    image = utils.pytorch_to_numpy(image)
    image = utils.convert_to_int(image)[0]
    cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_1.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    volume_rotated = trainer.transform.rotate_by_vector(volume, vector=torch.tensor([[1.0, 0.0, 0.0]], device=trainer.param.device))
    image = renderer.render(volume_rotated)
    image = utils.pytorch_to_numpy(image)
    image = utils.convert_to_int(image)[0]
    cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_2.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    volume_rotated = trainer.transform.rotate_by_vector(volume, vector=torch.tensor([[1.0, 1.0, 0.0]], device=trainer.param.device))
    image = renderer.render(volume_rotated)
    image = utils.pytorch_to_numpy(image)
    image = utils.convert_to_int(image)[0]
    cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_3.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    volume_rotated = trainer.transform.rotate_by_vector(volume, vector=torch.tensor([[1.0, 0.0, 1.0]], device=trainer.param.device))
    image = renderer.render(volume_rotated)
    image = utils.pytorch_to_numpy(image)
    image = utils.convert_to_int(image)[0]
    cv2.imwrite(os.path.join(trainer.dirs.tests, 'rendering_4.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


    print('[INFO] Rotation done')
