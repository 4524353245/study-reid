from __future__ import absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import models
from utils.losses import CrossEntropyLoss
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from utils.utils import Logger
from util.utils import AverageMeter,Logger,save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim

def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu : use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir,'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir,'log_test.txt'))
    print("========\nArgs:{}\n========".format(args))

    if use_gpu:
        print("Currently usimg GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_devices
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)  # 固定随机种子
    else:
        print("Currently using CPU")

    

