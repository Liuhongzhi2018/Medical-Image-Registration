import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
from torch.utils import tensorboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/core/logger.py def parse(args)
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/core/logger.py def dict_to_nonedict(opt)
    opt = Logger.dict_to_nonedict(opt)
    # writer = tensorboard.SummaryWriter(opt['path']["tb_logger"]+'/..')
    writer = tensorboard.SummaryWriter(opt['path']['experiments_root'])
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    phase = 'train'
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/data/__init__.py def create_dataset_3D(dataset_opt, phase)
    # train_set = Data.create_dataset_3D(dataset_opt, phase)
    train_set = Data.create_dataset_ACDC(dataset_opt, phase)
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/data/__init__.py def create_dataloader(dataset, dataset_opt, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')

    # model
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/__init__.py def create_model(opt)
    diffusion = Model.create_model(opt)
    print("Model Initialized")

    # Train
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/base_model.py self.begin_step = 0
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/model.py def feed_data(self, data)
            diffusion.feed_data(train_data)
            # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/model.py def optimize_parameters(self)
            diffusion.optimize_parameters()
            t = (time.time() - iter_start_time) / batchSize
            # log
            message = '(epoch: %d | iters: %d/%d | time: %.3f) ' % (current_epoch, (istep+1), training_iters, t)
            # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/model.py def get_current_log(self)
            errors = diffusion.get_current_log()
            for k, v in errors.items():
                message += '%s: %.6f ' % (k, v)
            print(message)
            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                writer.add_scalar("train/l_pix", logs['l_pix'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_sim", logs['l_sim'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_smt", logs['l_smt'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_tot", logs['l_tot'], (istep+1)*current_epoch)
              
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/model.py def save_network(self, epoch, iter_step)
            diffusion.save_network(current_epoch, current_step)