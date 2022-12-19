import warnings

warnings.filterwarnings('ignore')

import argparse
import logging
# import wandb
import sys
from argparse import Namespace

import torch

from pipeline.train import Train


def __init__(self):
        python_v = sys.version.split()[0]
        pytorch_v = torch.__version__
        cuda_s = torch.cuda.is_available()
        device = torch.cuda.current_device() if cuda_s else 'cpu'
        device_n = torch.cuda.get_device_name(device) if cuda_s else 'cpu'
        logging.info(f'python: {python_v} | pytorch: {pytorch_v} | gpu: {device_n if cuda_s else False}')
        self.device = device



def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--tag', default='Efficient_SAM_CE_alb', help='train process identifier')
    parser.add_argument('--folder', default='./images/', help='data root path')
    parser.add_argument('--num_classes', default=33, help='num_classes')
    parser.add_argument('--size', default=380, help='resize image to the specified size')
    parser.add_argument('--cache', default='cache', help='weights cache folder')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=25, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD Momentum.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Use 0.0 for no label smoothing.")

    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=32, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers number')
    
    # module opt
    parser.add_argument('--module', type=str, default=None, help='default None for efficientnet')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')

    # env INFO
    python_v = sys.version.split()[0]
    pytorch_v = torch.__version__
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    logging.info(f'python: {python_v} | pytorch: {pytorch_v} | gpu: {device_name if torch.cuda.is_available() else False}')

    # run train.
    train_process = Train(config, device)
    train_process.run()
