import argparse
import os
import torch
import shutil
import yaml
from attrdict import AttrDict
import torch.nn as nn
from torchvision import datasets, transforms


def make_dirs(param, root=None):
    class Dirs(object):
        pass

    dirs = Dirs()
    if root is None:
        dirs.root = os.getcwd()
    else:
        dirs.root = root

    run_name = '{}_{}_{}_{}_{}_{}_g2d{}_g3d{}_rec2d{}_rec3d{}_n_views{}_lr_g{}_lr_d{}_bs{}_{}'.format(
        param.job_id,
        param.name,
        param.mode,
        param.task,
        '_'.join(str(e) for e in param.data.use_classes),
        param.renderer.type,
        param.training.adversarial_term_lambda_2d,
        param.training.adversarial_term_lambda_3d,
        param.training.data_term_lambda_2d,
        param.training.data_term_lambda_3d,
        param.training.n_views,
        param.training.lr_g,
        param.training.lr_d,
        param.training.batch_size,
        param.training.view_sampling,
    )

    dirs.output = '{}/output/{}'.format(dirs.root, run_name)

    print('[INFO] log dir = {}'.format(run_name))

    dirs.stats = '{}/stats'.format(dirs.output)

    def mkdir_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdir_if_not_exists(dirs.stats)

    if param.tests.activate:
        dirs.tests = '{}/tests'.format(dirs.output)
        mkdir_if_not_exists(dirs.tests)

    return dirs


def load_config(config_path='scripts/configs/default_config.yaml'):
    parser = argparse.ArgumentParser(description='TODO.')
    parser.add_argument('--config_file', type=str, default=config_path,
                        help='Path to config file.')
    parser.add_argument('--job_id', type=int, default=1,
                        help='Path to config file.')

    args = parser.parse_args()
    path = args.config_file
    job_id = args.job_id

    with open(path, 'r') as f:
        config = yaml.load(f)

    # set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config['renderer']['type'] == 'emission_absorption':
        config['data']['n_channel_in'] = 4
        config['data']['n_channel_out_3d'] = 4
        config['data']['n_channel_out_2d'] = 4
    if config['renderer']['type'] == 'absorption_only' or config['renderer']['type'] == 'visual_hull':
        config['data']['n_channel_in'] = 1
        config['data']['n_channel_out_3d'] = 1
        config['data']['n_channel_out_2d'] = 1

    # make dict easier accessible by allowing for dot notation
    param = AttrDict(config)
    param._setattr('_sequence_type', list)

    param.job_id = job_id

    if param.device == 'cuda' and not torch.cuda.is_available():
        raise Exception('No GPU found, please use "cpu" as device')

    print('[INFO] Use {} {} device ({} devices are available)'.format(param.device, torch.cuda.current_device(), torch.cuda.device_count()))
    print('[INFO] Use pytorch {}'.format(torch.__version__))

    return param
