from torch.utils.data import Dataset
import os
import numpy as np
import scripts.renderer.transform as dt
import cv2
from scripts.utils.utils import normalize


class UCBDataset(Dataset):

    def __init__(self, param, mode):
        self.param = param
        self.transform = dt.Transform(self.param)

        print('[INFO] Setup dataset {}'.format(mode))

        self.path_dir = os.path.join(os.path.expanduser('~'), param.data.path_dir)
        self.mode = mode
        self.cube_len = param.data.cube_len
        self.n_channel = param.data.n_channel_in

        image_file = open('{}/images.txt'.format(self.path_dir), 'r')
        self.image_names = list(image_file)

        bounding_box_file = open('{}/bounding_boxes.txt'.format(self.path_dir), 'r')
        self.bounding_boxes = list(bounding_box_file)

        train_test_split_file = open('{}/train_test_split.txt'.format(self.path_dir), 'r')
        self.train_test_splits = list(train_test_split_file)

        train = 1

        if mode == "test":
            train = 0

        for i in range(len(self.train_test_splits)):
            if self.train_test_splits[i] == train:
                del self.image_names[i]
                del self.bounding_boxes[i]

        self.dataset_length = len(self.image_names)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):

        # idx = 4243

        image_name = self.image_names[idx].split(' ')[1].rstrip()
        id, x, y, width, height = self.bounding_boxes[idx].rstrip().split(' ')
        x = int(float(x))
        y = int(float(y))
        width = int(float(width))
        height = int(float(height))

        image_path = os.path.join(os.getcwd(), self.path_dir, 'images', image_name)
        segmentation_image_path = os.path.join(os.getcwd(), self.path_dir, 'segmentations', image_name).replace('jpg',
                                                                                                                'png')

        image_original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

        image = normalize(image_original)

        ## fixing annotation bugs
        if image_original.shape[0] < y + height:
            height += image_original.shape[0] - (y + height)

        if image_original.shape[1] < x + width:
            width += image_original.shape[1] - (x + width)

        segmentation_image = cv2.imread(segmentation_image_path, cv2.IMREAD_GRAYSCALE)
        segmentation_image = normalize(segmentation_image)

        segmentation_image = segmentation_image[..., np.newaxis]

        image = np.where(np.tile(segmentation_image, 3) >= 0.1, image, np.zeros_like(image))

        image = image[y:y + height, x:x + width, :]

        segmentation_image = segmentation_image[y:y + height, x:x + width]

        if height > width:
            image_square = np.zeros((height, height, 4))

            x_start = int((height - width) / 2.0)

            image_square[:, x_start:x_start + width, :3] = image
            image_square[:, x_start:x_start + width, [3]] = segmentation_image
        else:
            image_square = np.zeros((width, width, 4))
            y_start = int((width - height) / 2.0)
            image_square[y_start:y_start + height, :, :3] = image
            image_square[y_start:y_start + height, :, [3]] = segmentation_image

        image_square = cv2.copyMakeBorder(image_square, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)

        image = cv2.resize(image_square, (self.cube_len, self.cube_len), interpolation=cv2.INTER_CUBIC).astype(
            np.float32)

        image = np.clip(image, 0.0, 1.0)
        image = np.transpose(image, (2, 0, 1))
        image_ref = image
        image_random = image

        if self.param.renderer.type == 'visual_hull' or self.param.renderer.type == 'emission_absorption':
            alpha = image[3:4, :, :]
            alpha[alpha > 0.1] = 1.0
            alpha[alpha <= 0.1] = 0.0

            if self.param.renderer.type == 'visual_hull':
                image[3] = alpha
                image_random = image[[3]]
                image_ref = image[[3]]
        else:
            image = image[0:self.n_channel]
            image[0:3] = image[0:3] * image[3]

        return 0.0, image, image_ref, image_random, idx, image_path, str(idx), 0, 0
