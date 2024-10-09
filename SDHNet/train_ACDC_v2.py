import os
import sys
import argparse
import numpy as np
import random
import logging

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


def fetch_dataloader(args):
    if args.dataset == 'liver':
        train_dataset = datasets.LiverTrain(args)
    elif args.dataset == 'brain':
        train_dataset = datasets.BrainTrain(args)
    elif args.dataset == 'ACDC':
        train_dataset = datasets.ACDCTrain(args)
    else:
        print('Wrong Dataset')

    gpuargs = {'num_workers': 4, 'drop_last': True}
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False, sampler=train_sampler, **gpuargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False, **gpuargs)

    if args.local_rank == 0:
        print('Image pairs in training: %d' % len(train_dataset), file=args.files, flush=True)
    return train_loader


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.sum_freq = args.sum_freq

    def _print_training_status(self):
        metrics_data = ["{" + k + ":{:10.5f}".format(self.running_loss[k] / self.sum_freq) + "} "
                        for k in self.running_loss.keys()]
        training_str = "[Steps:{:9d}, Lr:{:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_lr()[0])
        print(training_str + "".join(metrics_data), file=args.files, flush=True)
        print(training_str + "".join(metrics_data))

        for key in self.running_loss:
            self.running_loss[key] = 0.0

    def push(self, metrics):
        self.total_steps = self.total_steps + 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] = self.running_loss[key] + metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            if args.local_rank == 0:
                self._print_training_status()
            self.running_loss = {}


def mask_class(label, value):
    return (torch.abs(label - value) < 0.5).float() * 255.0


def mask_metrics(seg1, seg2):
    sizes = np.prod(seg1.shape[1:])
    # seg1 = (seg1.view(-1, sizes) > 128).type(torch.float32)
    # seg2 = (seg2.view(-1, sizes) > 128).type(torch.float32)
    seg1 = (seg1.contiguous().view(-1, sizes) > 128).type(torch.float32)
    seg2 = (seg2.contiguous().view(-1, sizes) > 128).type(torch.float32)
    dice_score = 2.0 * torch.sum(seg1 * seg2, 1) / (torch.sum(seg1, 1) + torch.sum(seg2, 1))

    union = torch.sum(torch.max(seg1, seg2), 1)
    iden = (torch.ones(*union.shape) * 0.01).cuda()
    jacc_score = torch.sum(torch.min(seg1, seg2), 1) / torch.max(iden, union)

    return dice_score, jacc_score


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


def evaluate(args, Logging, eval_dataset, img_size, model, total_steps):
    Dice, Jacc, Jacb = [], [], []
    for i in range(len(eval_dataset)):
        fpath = eval_dataset[i][0]
        mov, fixed = eval_dataset[i][1][np.newaxis].cuda(), eval_dataset[i][2][np.newaxis].cuda()
        mov_label, fixed_label = eval_dataset[i][3][np.newaxis].cuda(), eval_dataset[i][4][np.newaxis].cuda()
        print(f"evaluate mov shape: {mov.shape} fixed shape: {fixed.shape}")
        print(f"evaluate mov_label shape: {mov_label.shape} fixed_label shape: {fixed_label.shape}")
        # evaluate mov shape: torch.Size([1, 1, 232, 288, 15]) fixed shape: torch.Size([1, 1, 232, 288, 15])
        # evaluate mov_label shape: torch.Size([1, 1, 232, 288, 15]) fixed_label shape: torch.Size([1, 1, 232, 288, 15])

        image1 = F.interpolate(mov, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        image2 = F.interpolate(fixed, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        label1 = F.interpolate(mov_label, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        label2 = F.interpolate(fixed_label, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        print(f"transpose mov shape: {image1.shape} fixed shape: {image2.shape}")
        print(f"transpose mov_label shape: {label1.shape} fixed_label shape: {label2.shape}")
        # transpose mov shape: torch.Size([1, 1, 80, 80, 80]) fixed shape: torch.Size([1, 1, 80, 80, 80])
        # transpose mov_label shape: torch.Size([1, 1, 80, 80, 80]) fixed_label shape: torch.Size([1, 1, 80, 80, 80])

        with torch.no_grad():
            # _, _, _, agg_flow, _ = model.module(image1, image2, augment=False)
            _, _, _, agg_flow, _ = model(image2, image1, augment=False)

            # print(f"eval_dataset seg_values: {eval_dataset.seg_values}")
            # eval_dataset seg_values: [0 1 2 3]
            image1_warped = warp3D()(image1, agg_flow)
            label1_warped = warp3D()(label1, agg_flow)

            jaccs = []
            dices = []

            for v in eval_dataset.seg_values:
                if v == 0: continue
                label2_c = mask_class(label2, v)
                label1_warped_c = warp3D()(mask_class(label1, v), agg_flow)

                class_dice, class_jacc = mask_metrics(label1_warped_c, label2_c)
                dices.append(class_dice)
                jaccs.append(class_jacc)

            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
            jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()
            
            Logging.info(f'Total_steps: {total_steps} Pair: {i:6d} dice:{dice:10.6f} jacc:{jacc:10.6f} jacb:{jacb:10.2f}')

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)

    dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
    jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
    jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))

    Logging.info(f'Total_steps {total_steps:12d} ---> \
                Dice:{dice_mean:10.6f}({dice_std:10.6f}) \
                Jacc:{jacc_mean:10.6f}({jacc_std:10.6f}) \
                Jacb:{jacb_mean:10.2f}({jacb_std:10.2f})')


def train(args):
        
    # Logger
    Logging = logging.getLogger()
    Logging.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.base_path, "output", 'train_' + args.dataset + '.log'))
    file_handler.setLevel(logging.INFO)
    Logging.addHandler(file_handler)
    Logging.addHandler(stdout_handler)
    # logger.info(f"Config: {args}")

    model = Framework(args)
    model.cuda()
    Logging.info(f"SDHNet model: {model}")
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader = fetch_dataloader(args)
    # img_size = (160, 192, 224)
    img_size = (80, 80, 80)

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, args)
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

            # image2_aug, affines, deforms, agg_flow, agg_flows = model(image1, image2)
            image1_aug, affines, deforms, agg_flow, agg_flows = model(image2, image1)

            loss, metrics = fetch_loss(affines, deforms, agg_flow, image1_aug, image2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps = total_steps + 1

            logger.push(metrics)
            Logging.info(f"Steps: {total_steps} / {args.num_steps} total loss: {loss:.4f} —— aff_loss: {metrics['aff_loss']:.4f} sim_loss: {metrics['sim_loss']:.4f} reg_loss: {metrics['reg_loss']:.4f}")

            if total_steps % args.val_freq == args.val_freq - 1:
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                torch.save(model.state_dict(), PATH)

                eval_dataset = datasets.ACDCTest(args)
                evaluate(args, Logging, eval_dataset, img_size, model, total_steps)

            if total_steps == args.num_steps:
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
    parser.add_argument('--dataset', type=str, default='ACDC', help='which dataset to use for training')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=100) # 2000
    parser.add_argument('--round', type=int, default=20000, help='number of batches per epoch')
    # parser.add_argument('--data_path', type=str, default='E:/Registration/Code/TMI2022/Github/Data_MRIBrain/')
    parser.add_argument('--data_path', type=str, default='/mnt/lhz/Github/Image_registration/SDHNet/images/')
    # parser.add_argument('--base_path', type=str, default='E:/Registration/Code/TMI2022/Github/')
    parser.add_argument('--base_path', type=str, default='/mnt/lhz/Github/Image_registration/SDHNet/')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    # args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    # args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset

    args.model_path = args.base_path + '/output/checkpoints_' + args.dataset
    args.eval_path = args.base_path + '/output/eval_' + args.dataset

    # dist.init_process_group(backend='nccl')

    if args.local_rank == 0:
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
    args.files = open(args.base_path + '/output/train_' + args.dataset + '.txt', 'a+')

    if args.local_rank == 0:
        print('Dataset: %s' % args.dataset, file=args.files, flush=True)
        print('Batch size: %s' % args.batch, file=args.files, flush=True)
        print('Step: %s' % args.num_steps, file=args.files, flush=True)
        print('Parallel GPU: %s' % args.nums_gpu, file=args.files, flush=True)

    train(args)
    args.files.close()
