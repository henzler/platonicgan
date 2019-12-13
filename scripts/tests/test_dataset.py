import os
from scripts.utils.utils import convert_to_int
import numpy as np
import cv2
import torch


def run_test_dataset(trainer):
    dataset_test = trainer.dataset_test
    renderer = trainer.renderer

    for idx in range(0, min(dataset_test.__len__(), 10)):

        data_sample = dataset_test[idx][0]

        data_sample = np.transpose(data_sample, (1, 2, 0))
        data_sample = np.squeeze(data_sample)
        data_sample = convert_to_int(data_sample)
        cv2.imwrite(os.path.join(trainer.dirs.tests, 'data_sample_{}_image.png'.format(idx)), data_sample)

        if not np.isscalar(dataset_test[idx][1]):

            volume_sample = dataset_test[idx][1]
            vector_sample = dataset_test[idx][2]

            volume_sample = volume_sample[np.newaxis, ...].to(trainer.param.device)
            vector_sample = torch.from_numpy(vector_sample[np.newaxis, ...]).float().to(trainer.param.device)

            volume_rotated = trainer.transform.rotate_by_vector(volume_sample.clone(), vector_sample)
            image = renderer.render(volume_rotated)
            image = image.detach().cpu().numpy()
            image = np.transpose(image[0], (1, 2, 0)).squeeze()
            image = cv2.resize(image, (trainer.param.data.cube_len, trainer.param.data.cube_len))
            image = convert_to_int(image)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            cv2.imwrite(os.path.join(trainer.dirs.tests, 'data_sample_{}_rendering.png'.format(idx)), image)

    print('[INFO] Dataset validation done')
