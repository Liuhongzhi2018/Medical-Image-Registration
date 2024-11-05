import os
import sys
import argparse
import numpy as np
import random
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

import core.datasets as datasets
import core.losses as losses
from core.utils.warp import warp3D
from core.framework import Framework
import SimpleITK as sitk
import scipy
import statistics
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, \
                                     binary_erosion,\
                                     generate_binary_structure


def fetch_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    milestones = [args.round*3, args.round*4, args.round*5]  # args.epoch == 5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    return optimizer, scheduler


# def fetch_loss(affines, deforms, agg_flow, image1, image2):
#     # affine loss
#     det = losses.det3x3(affines['A'])
#     det_loss = torch.sum((det - 1.0) ** 2) / 2

#     # I = torch.cuda.FloatTensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
#     I = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float, device='cuda')
#     eps = 1e-5
#     # epsI = torch.cuda.FloatTensor([[[eps * elem for elem in row] for row in Mat] for Mat in I])
#     epsI = torch.tensor([[[eps * elem for elem in row] for row in Mat] for Mat in I], dtype=torch.float, device='cuda')
#     C = torch.matmul(affines['A'].permute(0, 2, 1), affines['A']) + epsI
#     s1, s2, s3 = losses.elem_sym_polys_of_eigen_values(C)
#     ortho_loss = torch.sum(s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps))
#     aff_loss = 0.1 * det_loss + 0.1 * ortho_loss

#     # deform loss
#     image2_warped = warp3D()(image2, agg_flow)
#     # print(f"fetch_loss image1 shape: {image1.shape} image2_warped: {image2_warped.shape}")
#     # fetch_loss image1 shape: torch.Size([1, 1, 80, 80, 80]) image2_warped: torch.Size([1, 1, 80, 80, 80])
#     sim_loss = losses.similarity_loss(image1, image2_warped)

#     reg_loss = 0.0
#     for i in range(len(deforms)):
#         reg_loss = reg_loss + losses.regularize_loss(deforms[i]['flow'])

#     whole_loss = aff_loss + sim_loss + 0.5 * reg_loss

#     metrics = {
#         'aff_loss': aff_loss.item(),
#         'sim_loss': sim_loss.item(),
#         'reg_loss': reg_loss.item()
#     }

#     return whole_loss, metrics


def fetch_loss(affines, deforms, agg_flow, image1, image2):
    # affine loss
    det = losses.det3x3(affines['A'])
    det_loss = torch.sum((det - 1.0) ** 2) / 2

    # I = torch.cuda.FloatTensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    I = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float, device='cuda')
    eps = 1e-5
    # epsI = torch.cuda.FloatTensor([[[eps * elem for elem in row] for row in Mat] for Mat in I])
    epsI = torch.tensor([[[eps * elem for elem in row] for row in Mat] for Mat in I], dtype=torch.float, device='cuda')
    C = torch.matmul(affines['A'].permute(0, 2, 1), affines['A']) + epsI
    s1, s2, s3 = losses.elem_sym_polys_of_eigen_values(C)
    ortho_loss = torch.sum(s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps))
    aff_loss = 0.1 * det_loss + 0.1 * ortho_loss

    # deform loss
    # image2_warped = warp3D()(image2, agg_flow)
    image1_warped = warp3D()(image1, agg_flow)
    # print(f"fetch_loss image1 shape: {image1.shape} image2_warped: {image2_warped.shape}")
    # fetch_loss image1 shape: torch.Size([1, 1, 80, 80, 80]) image2_warped: torch.Size([1, 1, 80, 80, 80])
    
    # sim_loss = losses.similarity_loss(image1, image2_warped)
    sim_loss = losses.similarity_loss(image2, image1_warped)

    reg_loss = 0.0
    for i in range(len(deforms)):
        reg_loss = reg_loss + losses.regularize_loss(deforms[i]['flow'])

    whole_loss = aff_loss + sim_loss + 0.5 * reg_loss

    metrics = {
        'aff_loss': aff_loss.item(),
        'sim_loss': sim_loss.item(),
        'reg_loss': reg_loss.item()
    }

    return whole_loss, metrics


def fetch_dataloader(args, Logging):
    if args.dataset == 'liver':
        train_dataset = datasets.LiverTrain(args)
    elif args.dataset == 'brain':
        train_dataset = datasets.BrainTrain(args)
    elif args.dataset == 'OASIS':
        train_dataset = datasets.OASISTrain(args)
    else:
        print('Wrong Dataset')

    gpuargs = {'num_workers': 4, 'drop_last': True}
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False, sampler=train_sampler, **gpuargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False, **gpuargs)

    # if args.local_rank == 0:
    #     print('Image pairs in training: %d' % len(train_dataset), file=args.files, flush=True)
    Logging.info('Image pairs in training: %d' % len(train_dataset))
    # Image pairs in training: 100
    return train_loader


def save_samples(args, steps, image2_warped, label2_warped, agg_flow, img_path):
    # print(f"save_samples img_path {img_path}")
    # save_samples img_path /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient142/patient142_frame01.nii.gz
    index = img_path.split('/')[-1][:10]
    data_in = sitk.ReadImage(img_path)
    nii_array = sitk.GetArrayFromImage(data_in)
    # print(f"save_samples image2_warped {image2_warped.shape} label2_warped {label2_warped.shape} agg_flow {agg_flow.shape}")
    # save_samples image2_warped torch.Size([1, 1, 128, 128, 128]) label2_warped torch.Size([1, 1, 128, 128, 128]) agg_flow torch.Size([1, 3, 128, 128, 128])

    image2_warped = F.interpolate(image2_warped, size=nii_array.shape, mode='trilinear')
    label2_warped = F.interpolate(label2_warped, size=nii_array.shape, mode='trilinear')
    agg_flow = F.interpolate(agg_flow, size=nii_array.shape, mode='trilinear')
    # print(f"image2_warped {image2_warped.shape} label2_warped {label2_warped.shape} agg_flow {agg_flow.shape}")
    # image2_warped torch.Size([1, 1, 15, 288, 232]) label2_warped torch.Size([1, 1, 15, 288, 232]) agg_flow torch.Size([1, 3, 15, 288, 232])
    
    image2_warped_np = image2_warped.detach().cpu().numpy().squeeze()
    label2_warped_np = label2_warped.detach().cpu().numpy().squeeze()
    agg_flow_np = agg_flow.detach().cpu().numpy().squeeze().transpose(1, 2, 3, 0)
    # print(f"image2_warped_np {image2_warped_np.shape} label2_warped_np {label2_warped_np.shape} agg_flow_np {agg_flow_np.shape}")
    # image2_warped_np (15, 288, 232) label2_warped_np (15, 288, 232) agg_flow_np (15, 288, 232, 3)

    shape_img = data_in.GetSize()
    ED_origin = data_in.GetOrigin()
    ED_direction = data_in.GetDirection()
    ED_spacing = data_in.GetSpacing()
    # print(f"nii_array {nii_array.shape} shape_img {shape_img}")
    # nii_array (8, 256, 216) shape_img (216, 256, 8)

    savedSample_img = sitk.GetImageFromArray(image2_warped_np)
    savedSample_seg = sitk.GetImageFromArray(np.around(label2_warped_np))
    savedSample_defm = sitk.GetImageFromArray(agg_flow_np)
    
    savedSample_img.SetOrigin(ED_origin)
    savedSample_seg.SetOrigin(ED_origin)
    savedSample_defm.SetOrigin(ED_origin)
    
    savedSample_img.SetDirection(ED_direction)
    savedSample_seg.SetDirection(ED_direction)
    savedSample_defm.SetDirection(ED_direction)
    
    savedSample_img.SetSpacing(ED_spacing)
    savedSample_seg.SetSpacing(ED_spacing)
    savedSample_defm.SetSpacing(ED_spacing)
    
    warped_img_path = os.path.join(args.eval_path, index + '_ep' + str(steps) + '_warped_img.nii.gz')
    warped_seg_path = os.path.join(args.eval_path, index + '_ep' + str(steps) + '_warped_seg.nii.gz')
    warped_flow_path = os.path.join(args.eval_path, index + '_ep' + str(steps) + '_warped_deform.nii.gz')
    
    sitk.WriteImage(savedSample_img, warped_img_path)
    sitk.WriteImage(savedSample_seg, warped_seg_path)
    sitk.WriteImage(savedSample_defm, warped_flow_path)

    # print(f"warped_img_path {warped_img_path}")
    # print(f"warped_seg_path {warped_seg_path}")
    # print(f"warped_flow_path {warped_flow_path}")
    # warped_img_path /mnt/lhz/Github/Image_registration/SDHNet_checkpoints/ACDC_2024-11-04-22-16-53/eval/patient147_frame09_ep1_warped_img.nii.gz
    # warped_seg_path /mnt/lhz/Github/Image_registration/SDHNet_checkpoints/ACDC_2024-11-04-22-16-53/eval/patient147_frame09_ep1_warped_seg.nii.gz
    # warped_flow_path /mnt/lhz/Github/Image_registration/SDHNet_checkpoints/ACDC_2024-11-04-22-16-53/eval/patient147_frame09_ep1_warped_deform.nii.gz


class Logger:
    def __init__(self, model, scheduler, Logging, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.sum_freq = args.sum_freq
        self.Logging = Logging

    def _print_training_status(self):
        metrics_data = ["{" + k + ":{:10.5f}".format(self.running_loss[k] / self.sum_freq) + "} "
                        for k in self.running_loss.keys()]
        training_str = "[Steps:{:9d}, Lr:{:10.7f}] ".format(self.total_steps, self.scheduler.get_lr()[0])
        self.Logging.info(training_str + "".join(metrics_data))

        for key in self.running_loss:
            self.running_loss[key] = 0.0

    def push(self, metrics):
        self.total_steps = self.total_steps + 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] = self.running_loss[key] + metrics[key]

        # if self.total_steps % self.sum_freq == self.sum_freq - 1:
        #     if args.local_rank == 0:
        #         self._print_training_status()
        #     self.running_loss = {}
        self._print_training_status()
        self.running_loss = {}


def mask_class(label, value):
    return (torch.abs(label - value) < 0.5).float() * 255.0

def jacobian_det(flow):
    bias_d = np.array([0, 0, 1])
    bias_h = np.array([0, 1, 0])
    bias_w = np.array([1, 0, 0])

    volume_d = np.transpose(flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_d
    volume_h = np.transpose(flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_h
    volume_w = np.transpose(flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_w

    jacobian_det_volume = np.linalg.det(np.stack([volume_w, volume_h, volume_d], -1))
    jd = np.sum(jacobian_det_volume <= 0)
    return jd

def mask_metrics(seg1, seg2):
    sizes = np.prod(seg1.shape[1:])
    seg1 = (seg1.view(-1, sizes) > 128).type(torch.float32)
    seg2 = (seg2.view(-1, sizes) > 128).type(torch.float32)
    dice_score = 2.0 * torch.sum(seg1 * seg2, 1) / (torch.sum(seg1, 1) + torch.sum(seg2, 1))

    union = torch.sum(torch.max(seg1, seg2), 1)
    iden = (torch.ones(*union.shape) * 0.01).cuda()
    jacc_score = torch.sum(torch.min(seg1, seg2), 1) / torch.max(iden, union)

    return dice_score, jacc_score


def evaluate(args, Logging, eval_dataset, img_size, model, total_steps):
    Dice, Jacc, Jacb = [], [], []

    Logging.info('Image pairs in evaluation: %d' % len(eval_dataset))
    Logging.info('Evaluation steps: %s' % total_steps)
    # Image pairs in evaluation: 50
    # Evaluation steps: 1

    for i in range(len(eval_dataset)):
        fixed, mov = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
        fixed_label, mov_label = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()
        # print(f"evaluate mov shape: {mov.shape} fixed shape: {fixed.shape}")
        # print(f"evaluate mov_label shape: {mov_label.shape} fixed_label shape: {fixed_label.shape}")
        # evaluate mov shape: torch.Size([1, 1, 8, 256, 216]) fixed shape: torch.Size([1, 1, 8, 256, 216])
        # evaluate mov_label shape: torch.Size([1, 1, 8, 256, 216]) fixed_label shape: torch.Size([1, 1, 8, 256, 216])

        image1 = F.interpolate(fixed, size=img_size, mode='trilinear')
        image2 = F.interpolate(mov, size=img_size, mode='trilinear')
        label1 = F.interpolate(fixed_label, size=img_size, mode='trilinear')
        label2 = F.interpolate(mov_label, size=img_size, mode='trilinear')
        # print(f"transpose fixed shape: {image1.shape} mov shape: {image2.shape}")
        # print(f"transpose fixed_label shape: {label1.shape} mov_label shape: {label2.shape}")
        # transpose fixed shape: torch.Size([1, 1, 128, 128, 128]) mov shape: torch.Size([1, 1, 128, 128, 128])
        # transpose fixed_label shape: torch.Size([1, 1, 128, 128, 128]) mov_label shape: torch.Size([1, 1, 128, 128, 128])

        with torch.no_grad():
            _, _, _, agg_flow, _ = model(image1, image2, augment=False)
            image2_warped = warp3D()(image2, agg_flow)
            label2_warped = warp3D()(label2, agg_flow)

        jaccs = []
        dices = []

        for v in eval_dataset.seg_values:
            label1_fixed = mask_class(label1, v)
            # label2_warped = warp3D()(mask_class(label2, v), agg_flow)
            label2_in = mask_class(label2, v)
            label2_v = warp3D()(label2_in, agg_flow)
            # print(f"evaluate label1_fixed {label1_fixed.shape} label2_warped {label2_v.shape}")
            # evaluate label1_fixed torch.Size([1, 1, 128, 128, 128]) label2_warped torch.Size([1, 1, 128, 128, 128])
            # print(f"evaluate label1_fixed max {label1_fixed.max()} min {label1_fixed.min()} label2_in max {label2_in.max()} min {label2_in.min()}")
            # evaluate label1_fixed max 255.0 min 0.0 label2_in max 255.0 min 0.0
            # print(f"evaluate label2_v max {label2_v.max()} min {label2_v.min()}")
            # evaluate label2_v max 255.0000457763672 min 0.0
            class_dice, class_jacc = mask_metrics(label1_fixed, label2_v)

            dices.append(class_dice)
            jaccs.append(class_jacc)

        jacb = jacobian_det(agg_flow.cpu().numpy()[0])

        dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
        jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()
        
        Logging.info('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   jacb:{:10.2f}'.
                    format(i, dice, jacc, jacb))

        save_samples(args, total_steps, image2_warped, label2_warped, agg_flow, eval_dataset[i][4])

        Dice.append(dice)
        Jacc.append(jacc)
        Jacb.append(jacb)
            
    dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
    jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
    jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))

    Logging.info('Summary --->  '
                    'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                    'Jacb:{:10.2f}({:10.2f})'
                    .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std))

    Logging.info('Step{:12d} --->  '
                    'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                    'Jacb:{:10.2f}({:10.2f})'
                    .format(total_steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std))


def train(args, Logging):

    model = Framework(args)
    model.cuda()
    Logging.info(f"SDHNet model: {model}")
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader = fetch_dataloader(args, Logging)
    # img_size = (232, 256, 10)
    # img_size = (64, 64, 64)
    img_size = (128, 128, 128)

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, Logging, args)
    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            model.train()
            # image1, image2 = [x.cuda(non_blocking=True) for x in data_blob]
            image1_data, image2_data = [x.cuda() for x in data_blob]
            optimizer.zero_grad()
            # print(f"train image1 shape: {image1_data.shape} image2 shape: {image2_data.shape}")
            # train image1 shape: torch.Size([1, 1, 428, 512, 8]) image2 shape: torch.Size([1, 1, 428, 512, 8])
            image1 = F.interpolate(image1_data, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
            image2 = F.interpolate(image2_data, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
            # print(f"transpose image1 shape: {image1.shape} image2 shape: {image2.shape}")
            # transpose image1 shape: torch.Size([1, 1, 8, 512, 428]) image2 shape: torch.Size([1, 1, 8, 512, 428])
            # transpose image1 shape: torch.Size([1, 1, 80, 80, 80]) image2 shape: torch.Size([1, 1, 80, 80, 80])

            image2_aug, affines, deforms, agg_flow, agg_flows = model(image1, image2)
            # image1_aug, affines, deforms, agg_flow, agg_flows = model(image2, image1)

            # loss, metrics = fetch_loss(affines, deforms, agg_flow, image1_aug, image2)
            loss, metrics = fetch_loss(affines, deforms, agg_flow, image1, image2_aug)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps = total_steps + 1

            logger.push(metrics)
            # Logging.info(f"Steps: {total_steps} / {args.num_steps} total loss: {loss:.4f} —— aff_loss: {metrics['aff_loss']:.4f} sim_loss: {metrics['sim_loss']:.4f} reg_loss: {metrics['reg_loss']:.4f}")

            # if total_steps % args.val_freq == args.val_freq - 1:
            # if total_steps % args.val_freq == 0:
            if total_steps % args.val_freq == 1:
                model.eval()
                eval_dataset = datasets.OASISTest(args)
                evaluate(args, Logging, eval_dataset, img_size, model, total_steps)

                # PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps == args.num_steps + 1:
                should_keep_training = False
                break

    # PATH = args.model_path + '/%s.pth' % args.name
    PATH = args.model_path + '/%s_final.pth' % args.name
    torch.save(model.state_dict(), PATH)

    # return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='SDHNet', help='name your experiment')
    # parser.add_argument('--dataset', type=str, default='brain', help='which dataset to use for training')
    parser.add_argument('--dataset', type=str, default='OASIS', help='which dataset to use for training')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=50000) # 2000
    parser.add_argument('--round', type=int, default=20000, help='number of batches per epoch')
    # parser.add_argument('--data_path', type=str, default='E:/Registration/Code/TMI2022/Github/Data_MRIBrain/')
    parser.add_argument('--data_path', type=str, default='/mnt/lhz/Github/Image_registration/SDHNet/images/')
    # parser.add_argument('--base_path', type=str, default='E:/Registration/Code/TMI2022/Github/')
    parser.add_argument('--base_path', type=str, default='/mnt/lhz/Github/Image_registration/SDHNet_checkpoints/')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    # args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    # args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset

    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    # args.model_path = args.base_path + '/output/checkpoints_' + args.dataset
    args.model_path = os.path.join(args.base_path, args.dataset + '_' + curr_time, 'checkpoints')
    # args.eval_path = args.base_path + '/output/eval_' + args.dataset
    args.eval_path = os.path.join(args.base_path, args.dataset + '_' + curr_time, 'eval')

    # dist.init_process_group(backend='nccl')

    if args.local_rank == 0:
        os.makedirs(args.base_path, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
        os.makedirs(args.eval_path, exist_ok=True)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    args.nums_gpu = torch.cuda.device_count()
    args.batch = args.batch
    args.num_steps = args.epoch * args.round
    # args.files = open(args.base_path + args.name + '/output/train_' + args.dataset + '.txt', 'a+')
    # args.files = open(args.base_path + '/output/train_' + args.dataset + '.txt', 'a+')

    # if args.local_rank == 0:
    #     print('Dataset: %s' % args.dataset, file=args.files, flush=True)
    #     print('Batch size: %s' % args.batch, file=args.files, flush=True)
    #     print('Step: %s' % args.num_steps, file=args.files, flush=True)
    #     print('Parallel GPU: %s' % args.nums_gpu, file=args.files, flush=True)

    # Logger
    Logging = logging.getLogger()
    Logging.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.model_path, 'train_' + args.dataset + '.log'))
    file_handler.setLevel(logging.INFO)
    Logging.addHandler(file_handler)
    Logging.addHandler(stdout_handler)
    # Logging.info(f"Config: {args}")
    Logging.info('Dataset: %s' % args.dataset)
    Logging.info('Batch size: %s' % args.batch)
    Logging.info('Step: %s' % args.num_steps)
    Logging.info('Parallel GPU: %s' % args.nums_gpu)

    train(args, Logging)
    # args.files.close()
