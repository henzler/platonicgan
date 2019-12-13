from scripts.data import image, caltech_ucsd_birds

dataset_dict = {
    'image': image.ImageDataset,
    'ucb':  caltech_ucsd_birds.UCBDataset
}