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
import torch.nn.functional as F
import core.metrics as Metrics
import SimpleITK as sitk
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/core/logger.py def parse(args)
    opt = Logger.parse(args)
    print(f"opt: {opt}")
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
        sample_data = 0
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            sample_data = train_data
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

        # sample
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py
            diffusion.test_registration()
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py 
            # def get_current_registration
            visuals = diffusion.get_current_registration()
            # print(visuals['contF'].shape)
            defm_frames_visual = visuals['contD'].squeeze(0).numpy().transpose(0, 2, 3, 1)
            flow_frames = visuals['contF'].numpy().transpose(0, 3, 4, 2, 1)
            flow_frames_ES = flow_frames[-1]
            # print(f"defm_frames_visual shape: {defm_frames_visual.shape} flow_frames_ES shape: {flow_frames_ES.shape}")
            # defm_frames_visual shape: (1, 128, 128, 32) flow_frames_ES shape: (128, 128, 32, 3)
            sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/core/metrics.py def transform_grid(dDepth, dHeight, dWidth)
            sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
            nb, nc, nd, nh, nw = sflow.shape
            segflow = torch.FloatTensor(sflow.shape).zero_()
            segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
            segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
            segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
            origin_seg = sample_data['MS'].squeeze()
            origin_seg = origin_seg.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            regist_seg = F.grid_sample(origin_seg.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)),
                                    mode='nearest')
            regist_seg_ = regist_seg.permute(0, 1, 3, 4, 2)
            regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
            label_seg = sample_data['FS'][0].cpu().numpy()
            origin_seg = sample_data['MS'][0].cpu().numpy()
            origin_ED = sample_data['imageED'][0].cpu().numpy()
            origin_ES = sample_data['imageES'][0].cpu().numpy()
            defm_frames_data = defm_frames_visual[0]
            # defm_frames_data = np.rot90(defm_frames_data, k=-2, axes=(0, 1))
            # print(f"defm_frames_data {defm_frames_data.shape} origin_ED shape {origin_ED.shape} origin_ES shape {origin_ES.shape} origin_seg shape {origin_seg.shape} regist_seg shape {regist_seg.shape} label_seg shape {label_seg.shape}")
            # origin_seg shape (128, 128, 32) regist_seg shape (128, 128, 32) label_seg shape (128, 128, 32)
            
            data_in = sitk.ReadImage(sample_data['Path'][0])
            # ED_origin = data_in.GetOrigin()[:-1]
            # ED_direction = np.array(list(data_in.GetDirection())).reshape(4, 4)[:3, :3].flatten()
            # ED_spacing = data_in.GetSpacing()[:-1]
            ED_origin = data_in.GetOrigin()
            ED_direction = data_in.GetDirection()
            ED_spacing = data_in.GetSpacing()
            ED_shape = data_in.GetSize()
            # print(f"Origin: {ED_origin} Direction: {ED_direction} Spacing: {ED_spacing} Size: {ED_shape}")
            # Origin: (0.0, 0.0, 0.0) Direction: [1. 0. 0. 0. 1. 0. 0. 0. 1.] Spacing: (1.6796875, 1.6796875, 10.0)
            # Origin: (-116.78947448730469, 98.10527038574219, -586.900146484375) 
            # Direction: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
            # Spacing: (0.41015625, 0.41015625, 0.4000001549720764) 
            # Size: (512, 512, 308)
            
            origin_ED1 = origin_ED.transpose(2, 1, 0)
            origin_ES1 = origin_ES.transpose(2, 1, 0)
            origin_seg1 = origin_seg.transpose(2, 1, 0)
            regist_seg1 = regist_seg.transpose(2, 1, 0)
            label_seg1 = label_seg.transpose(2, 1, 0)
            defm_frames_data1 = defm_frames_data.transpose(2, 1, 0)
            # print(f"Transpose origin_ED1 shape {origin_ED1.shape} origin_seg shape {origin_seg1.shape} origin_ES1 shape {origin_ES1.shape} label_seg shape {label_seg1.shape} defm_frames_data1 shape {defm_frames_data1.shape} regist_seg shape {regist_seg1.shape}")

            savedSample_ED = sitk.GetImageFromArray(origin_ED1)
            savedSample_ES = sitk.GetImageFromArray(origin_ES1)
            savedSample_origin = sitk.GetImageFromArray(origin_seg1)
            savedSample_regist = sitk.GetImageFromArray(regist_seg1)
            savedSample_label = sitk.GetImageFromArray(label_seg1)
            savedSample_defm = sitk.GetImageFromArray(defm_frames_data1)
            # print(f"train_data['Path']: {train_data['Path'][0]}")
            
            savedSample_ED.SetOrigin(ED_origin)
            savedSample_ES.SetOrigin(ED_origin)
            savedSample_origin.SetOrigin(ED_origin)
            savedSample_regist.SetOrigin(ED_origin)
            savedSample_label.SetOrigin(ED_origin)
            savedSample_defm.SetOrigin(ED_origin)
            
            savedSample_ED.SetDirection(ED_direction)
            savedSample_ES.SetDirection(ED_direction)
            savedSample_origin.SetDirection(ED_direction)
            savedSample_regist.SetDirection(ED_direction)
            savedSample_label.SetDirection(ED_direction)
            savedSample_defm.SetDirection(ED_direction)
            
            savedSample_ED.SetSpacing(ED_spacing)
            savedSample_ES.SetSpacing(ED_spacing)
            savedSample_origin.SetSpacing(ED_spacing)
            savedSample_regist.SetSpacing(ED_spacing)
            savedSample_label.SetSpacing(ED_spacing)
            savedSample_defm.SetSpacing(ED_spacing)

            # print(f"Origin: {ED_origin} Direction: {ED_direction} Spacing: {ED_spacing}")
            # sitk.WriteImage(savedSample_origin, os.path.join(opt['path']['experiments_root'], "sample_" + str(current_epoch)+"_origin.nii.gz"))
            # sitk.WriteImage(savedSample_regist, os.path.join(opt['path']['experiments_root'],"sample_" + str(current_epoch)+"_regist.nii.gz"))
            # sitk.WriteImage(savedSample_label, os.path.join(opt['path']['experiments_root'],"sample_" + str(current_epoch)+"_label.nii.gz"))
            # shutil.copyfile(train_data['Path'][0], os.path.join(opt['path']['experiments_root'], "sample_image_latest.nii.gz"))
            sitk.WriteImage(savedSample_ED, os.path.join(opt['path']['experiments_root'], "sample_moving_ep"+str(current_epoch)+".nii.gz"))
            sitk.WriteImage(savedSample_ES, os.path.join(opt['path']['experiments_root'], "sample_fixed_ep"+str(current_epoch)+".nii.gz"))
            sitk.WriteImage(savedSample_origin, os.path.join(opt['path']['experiments_root'], "sample_moving_label_ep"+str(current_epoch)+".nii.gz"))
            sitk.WriteImage(savedSample_label, os.path.join(opt['path']['experiments_root'],"sample_fixed_label_ep"+str(current_epoch)+".nii.gz"))
            sitk.WriteImage(savedSample_regist, os.path.join(opt['path']['experiments_root'],"sample_warped_label_ep"+str(current_epoch)+".nii.gz"))
            sitk.WriteImage(savedSample_defm, os.path.join(opt['path']['experiments_root'],"sample_warped_ep"+str(current_epoch)+".nii.gz"))

              
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/model/model.py def save_network(self, epoch, iter_step)
            diffusion.save_network(current_epoch, current_step)