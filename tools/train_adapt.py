# -*- encoding: utf-8 -*-
#Time        :2021/01/04 16:58:26
#Author      :Chen
#FileName    :train_contrast.py
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
from dataset.polyp_dataset import Contrast_Polyp, Polyp
from models.deeplab import Deeplab
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 2
NUM_WORKERS = 4
POWER = 0.9
INPUT_SIZE = (256, 256)
SOURCE_DATA = '/home/cyang/SFDA/data/EndoScene'
TRAIN_SOURCE_LIST = '/home/cyang/SFDA/dataset/EndoScene_list/train.lst'
TEST_SOURCE_LIST = '/home/cyang/SFDA/dataset/EndoScene_list/test.lst'

TARGET_DATA = '/home/cyang/SFDA/data/Etislarib'
TRAIN_TARGET_LIST = '/home/cyang/SFDA/dataset/Etislarib_list/train_fold4.lst'
TEST_TARGET_LIST = '/home/cyang/SFDA/dataset/Etislarib_list/test_fold4.lst'

LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 150
VALID_STEPS = 100
GPU = '7'
FOLD = 'fold4'
TARGET_MODE = 'gt'
RESTORE_FROM = '/home/cyang53/CED/Ours/SFDA-TMI/checkpoint/Endo_best.pth'
SNAPSHOT_DIR = '/home/cyang53/CED/Ours/SFDA-TMI/checkpoint/'
SAVE_RESULT = False
RANDOM_MIRROR = True
IS_ADABN = False
IS_INVERSE = True
IS_PSEUDO = True
IS_CONTRAST = True

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, length, power=0.9):
    lr = lr_poly(args.learning_rate, i_iter, NUM_STEPS * length, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


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
    parser.add_argument("--source-train", type=str, default=TRAIN_SOURCE_LIST,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--source-test", type=str, default=TEST_SOURCE_LIST,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is_adabn", type=bool, default=IS_ADABN,
                        help="Whether to apply test mean and var rather than running.")
    parser.add_argument("--data-dir-target", type=str, default=TARGET_DATA,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--target-train", type=str, default=TRAIN_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--target-test", type=str, default=TEST_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--w-contrast", type=float, default=0.01,
                        help="Weight of contrast loss.")
    parser.add_argument("--w-distill", type=float, default=0.1,
                        help="Weight of distill loss.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--valid-steps", type=int, default=VALID_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", type=bool, default=RANDOM_MIRROR,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=SAVE_RESULT,
                        help="Whether to save the predictions.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--fold", type=str, default=FOLD,
                        help="choose gpu device.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--is-inverse", type=bool, default=IS_INVERSE,
                        help="use inversed images.")
    parser.add_argument("--is-pseudo", type=bool, default=IS_PSEUDO,
                        help="use pseudo labels.")
    parser.add_argument("--is-contrast", type=bool, default=IS_CONTRAST,
                        help="apply contrast loss.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    contrast_dataset = Contrast_Polyp(root=args.data_dir_target,
                         data_dir=args.target_train, mode='train', max_iter=None, is_mirror=args.random_mirror, is_pseudo=args.is_pseudo, is_inverse=args.is_inverse)
    test_target = Polyp(root=args.data_dir_target,
                        data_dir=args.target_test, mode='test', is_mirror=False, is_inverse= args.is_inverse)


    contrast_loader_target = torch.utils.data.DataLoader(
        contrast_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_target = torch.utils.data.DataLoader(
        test_target, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = Deeplab(1, pretrained=True, contrast=args.is_contrast).cuda()
    #model.load_state_dict(torch.load(args.restore_from))

    optimizer = torch.optim.SGD(
        model.optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    source_criterion = DiceLoss()
    
    # Eq. (4): MSE loss to calculate feature discrepancy
    distill_criterion = nn.MSELoss()
    
    #target_criterion = criterion.BCELoss()
    Best_dice = 0
    for epoch in range(args.num_steps):
        seg_loss = 0
        dis_loss = 0
        con_loss = 0
        tic = time.time()
        model.train()
        for i_iter, batch in enumerate(contrast_loader_target):
            data, weight, name = batch
            source_image = data['source_image']
            source_image_rotate = data['source_image_rotate']
            target_image = data['target_image']
            target_image_rotate = data['target_image_rotate']
            label = data['label']
            label_rotate = data['label_rotate']

            source_image = Variable(source_image).cuda()
            source_image_rotate = Variable(source_image_rotate).cuda()
            target_image = Variable(target_image).cuda()
            target_image_rotate = Variable(target_image_rotate).cuda()


            label = Variable(label).cuda()
            label = label.unsqueeze(1)
            label_rotate = Variable(label_rotate).cuda()
            label_rotate = label_rotate.unsqueeze(1)
            weight = Variable(weight).cuda()
            source_feature, source_output = model(source_image)
            source_feature_rotate, source_output_rotate = model(source_image_rotate)
            target_feature, target_output = model(target_image)
            target_feature_rotate, target_output_rotate = model(target_image_rotate)
            
            # Eq. (5): domain distillation loss to learn structure-wise knowledge
            loss_distill = distill_criterion(distill_criterion(source_feature, source_feature_rotate), distill_criterion(target_feature, target_feature_rotate))
               
            # Eq. (6): domain contrastive loss to narrow down the domain gap by self-supervised paradigm
            loss_contrast = distill_criterion(source_feature, target_feature) + distill_criterion(distill_criterion(source_feature, target_feature_rotate), distill_criterion(source_feature, source_feature_rotate))
            
            # Eq. (7): compact-aware domain consistency loss to achieve output-level adaptation
            loss_pseudo = source_criterion(source_output, label, weight=weight) + source_criterion(target_output, label, weight=weight) + source_criterion(source_output_rotate, label_rotate, weight=weight) + source_criterion(target_output_rotate, label_rotate, weight=weight)
            
            loss_total = loss_pseudo + args.w_distill * loss_distill + args.w_contrast * loss_contrast
                
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()
            seg_loss += loss_pseudo.item()
            dis_loss += loss_distill.item()
            con_loss += loss_contrast.item()
            lr = adjust_learning_rate(optimizer=optimizer, i_iter=i_iter + epoch * contrast_dataset.__len__(
            ) / args.batch_size, length=contrast_dataset.__len__() / args.batch_size)
            #lr = args.learning_rate
        batch_time = time.time() - tic
        print('Epoch: [{}/{}], Time: {:.2f}, '
              'lr: {:.6f}, Seg Loss: {:.6f}, Dis Loss: {:.6f}, Con Loss: {:.6f}' .format(
                  epoch, args.num_steps, batch_time, lr, seg_loss, dis_loss, con_loss))
        # begin test on target domain
        dice = test(model, test_loader_target, args)

    torch.save(model.state_dict(), '/home/cyang/SFDA/checkpoint/FSM_last.pth')


if __name__ == '__main__':
    main()
