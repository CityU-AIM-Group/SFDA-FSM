# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:17:12
#Author      :Chen
#FileName    :test.py
#Version     :1.0

import torch
import _init_paths
import argparse
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.metric import Metrics, setup_seed, test
from utils.metric import evaluate
from utils.loss import BceDiceLoss, BCELoss, DiceLoss
from dataset.polyp_dataset import Polyp
from models.deeplab import Deeplab
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


BATCH_SIZE = 8
NUM_WORKERS = 4
POWER = 0.9
INPUT_SIZE = (256, 256)
SOURCE_DATA = '/home/cyang/SFDA/data/EndoScene'
TEST_SOURCE_LIST = '/home/cyang/SFDA/dataset/EndoScene_list/test.lst'

TARGET_DATA = '/home/cyang/SFDA/data/Etislarib'
TRAIN_TARGET_LIST = '/home/cyang/SFDA/dataset/Etislarib_list/train_fold4.lst'
TEST_TARGET_LIST = '/home/cyang/SFDA/dataset/Etislarib_list/test_fold4.lst'
ADABN_LIST = '/home/cyang/SFDA/dataset/Etislarib_list/adabn_fold4.lst'

NUM_CLASSES = 1
GPU = '7'
TRANSFER = True
TARGET_MODE = 'gt'
RESTORE_FROM = '/home/cyang/SFDA/checkpoint/Endo_Pro.pth'
SNAPSHOT_DIR = '/home/cyang/SFDA/checkpoint/'
SAVE_RESULT = True
RANDOM_MIRROR = False
IS_ADABN = True
IS_INVERSE = False
IS_CONTRAST = False
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=SOURCE_DATA,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--source-test", type=str, default=TEST_SOURCE_LIST,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is_adabn", type=bool, default=IS_ADABN,
                        help="Whether to apply test mean and var rather than running.")
    parser.add_argument("--is_inverse", type=bool, default=IS_INVERSE,
                        help="Whether to use inversed images.")
    parser.add_argument("--data-dir-target", type=str, default=TARGET_DATA,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--target-train", type=str, default=TRAIN_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--target-test", type=str, default=TEST_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--random-mirror", type=bool, default=RANDOM_MIRROR,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=SAVE_RESULT,
                        help="Whether to save the predictions.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--transfer", type=bool, default=TRANSFER,
                        help="choose gpu device.")
    parser.add_argument("--is-contrast", type=bool, default=IS_CONTRAST,
                        help="apply contrast loss.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_source = Polyp(root=args.data_dir,
                        data_dir=args.source_test, mode='test', is_mirror=args.random_mirror)

    test_target = Polyp(root=args.data_dir_target,
                        data_dir=args.target_test, mode='test', is_mirror=args.random_mirror, is_inverse=args.is_inverse)

    pseudo_target = Polyp(root=args.data_dir_target,
                        data_dir=args.target_train, mode='test', is_mirror=args.random_mirror, is_inverse=args.is_inverse)
    adabn_target = Polyp(root=args.data_dir_target,
                        data_dir=ADABN_LIST, mode='test', is_mirror=args.random_mirror)

    adabn_loader = torch.utils.data.DataLoader(
        adabn_target, batch_size=8, shuffle=True, num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_source, batch_size=1, shuffle=False, num_workers=args.num_workers)

    pseudo_loader = torch.utils.data.DataLoader(
        pseudo_target, batch_size=1, shuffle=False, num_workers=args.num_workers)

    test_loader_target = torch.utils.data.DataLoader(
        test_target, batch_size=1, shuffle=False, num_workers=args.num_workers)
    

    model = Deeplab(1, pretrained=False).cuda()
    model.load_state_dict(torch.load(args.restore_from))
    dice = test(model, test_loader_target, args, adabn_loader)

if __name__ == '__main__':
    main()
