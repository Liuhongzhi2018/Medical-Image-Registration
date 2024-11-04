import os
import argparse
import numpy as np
import random
import sys
import time
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
from core.framework import Framework_Teacher, augmentation
from eval import evaluate


def make_dirs(args):
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path + "/teacher")

    if not os.path.isdir(args.eval_path):
        os.makedirs(args.eval_path + "/teacher")

def count_parameters(model, Logging, type):
    # print("\n----------" + type.title() + "-Net Params----------", file=args.files, flush=True)
    # print("\n----------" + type.title() + "-Net Params----------")
    Logging.info("\n----------" + type.title() + "-Net Params----------")
    # print('Whole parameters: %d'
    #       % sum(p.numel() for p in model.parameters() if p.requires_grad),
    #       file=args.files, flush=True)
    # print('Whole parameters: %d'
    #       % sum(p.numel() for p in model.parameters() if p.requires_grad))
    Logging.info('Whole parameters: %d'
            % sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print('Affine parameters: %d'
    #       % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad),
    #       file=args.files, flush=True)
    # print('Affine parameters: %d'
    #       % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad))
    Logging.info('Affine parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad))
    # print('Deform parameters: %d'
    #       % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad),
    #       file=args.files, flush=True)
    # print('Deform parameters: %d'
    #       % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad))
    Logging.info('Deform parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad))
    # print("----------" + type.title() + "-Net Params----------\n", file=args.files, flush=True)
    # print("----------" + type.title() + "-Net Params----------\n")
    Logging.info("----------" + type.title() + "-Net Params----------\n")

def mask_class(label, value):
    return (torch.abs(label - value) < 0.5).float() * 255.0

def mask_metrics(seg1, seg2):
    sizes = np.prod(seg1.shape[1:])
    seg1 = (seg1.view(-1, sizes) > 128).type(torch.float32)
    seg2 = (seg2.view(-1, sizes) > 128).type(torch.float32)
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

def fetch_loss(affines, deforms, agg_flow, image1, image2):

    affines = None

    # deform loss
    image2_warped = warp3D()(image2, agg_flow)
    sim_loss = losses.similarity_loss(image1, image2_warped)

    reg_loss = 0.0
    for i in range(len(deforms['deform'])):
        reg_loss = reg_loss + losses.regularize_loss(deforms['deform'][i])

    whole_loss = sim_loss + 1 * reg_loss

    metrics = {
        'sim_loss': sim_loss.item(),
        'reg_loss': reg_loss.item()
    }

    return whole_loss, metrics


def fetch_dataloader(args):
    if args.dataset == 'ACDC':
        train_dataset = datasets.ACDCTrain(args)
    elif args.dataset == 'LPBA':
        train_dataset = datasets.LPBATrain(args)
    elif args.dataset == 'OASIS':
        train_dataset = datasets.OASISTrain(args)
    elif args.dataset == 'OAIZIB':
        train_dataset = datasets.OAIZIBTrain(args)
    else:
        print('Wrong Dataset')

    gpuargs = {'num_workers': 4, 'drop_last': True}
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False,
    #                           sampler=train_sampler, **gpuargs)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch, 
                              pin_memory=True, 
                              shuffle=False, **gpuargs)

    # if args.local_rank == 0:
    #     print('Image pairs in training: %d' % len(train_dataset), file=args.files, flush=True)
    
    print('Image pairs in training: %d' % len(train_dataset))
    # Image pairs in training: 394
    return train_loader


def fetch_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    milestones = [args.round * 3, args.round * 4, args.round * 5]  # args.epoch == 5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    return optimizer, scheduler


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
        # print(training_str + "".join(metrics_data), file=args.files, flush=True)
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
        # if self.total_steps % self.sum_freq == 0:
            # if args.local_rank == 0:
            #     self._print_training_status()
        self._print_training_status()
        self.running_loss = {}


def evaluate_ACDC(args, model, steps, Logging, type):
    # for datas in args.dataset_val:
    #     eval_path = join(args.eval_path, type, datas)
    #     if not os.path.isdir(eval_path):
    #         os.makedirs(eval_path)
    #     file_sum = join(eval_path, datas + '.txt')
    #     file = join(eval_path, datas + '_' + str(steps) + '.txt')
    #     f = open(file, 'a+')
    #     g = open(file_sum, 'a+')

    Dice, Jacc, Jacb, Time = [], [], [], []
    eval_dataset = datasets.ACDCTest(args)
    # if args.local_rank == 0:
    #     print('Dataset in evaluation: %s' % datas, file=f, flush=True)
    #     print('Dataset in evaluation: %s' % datas)
    #     print('Image pairs in evaluation: %d' % len(eval_dataset), file=f, flush=True)
    #     print('Image pairs in evaluation: %d' % len(eval_dataset))
    #     print('Evaluation steps: %s' % steps, file=f, flush=True)
    #     print('Evaluation steps: %s' % steps)
    #     print('Model Type: %s' % type, file=f, flush=True)
    #     print('Model Type: %s' % type)
    Logging.info('Image pairs in evaluation: %d' % len(eval_dataset))
    Logging.info('Evaluation steps: %s' % steps)
    Logging.info('Model Type: %s' % type)
    # count_parameters(model, Logging, type="teacher")

    for i in range(len(eval_dataset)):
        # image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
        # label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()
        # image1-fixed image2-moving
        image1, image2  = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
        label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()
        # print(f"evaluate_ACDC image1 {image1.shape} image2 {image2.shape}")
        # evaluate_ACDC image1 torch.Size([1, 1, 9, 256, 214]) image2 torch.Size([1, 1, 9, 256, 214])
        # print(f"evaluate_ACDC image1 max {image1.max()} min {image1.min()} image2 max {image2.max()} min {image2.min()}")
        # evaluate_ACDC image1 max 255.0 min 0.0 image2 max 255.0 min 0.0
        # print(f"evaluate_ACDC label1 {label1.shape} label2 {label2.shape}")
        # evaluate_ACDC label1 torch.Size([1, 1, 9, 256, 214]) label2 torch.Size([1, 1, 9, 256, 214])
        # print(f"evaluate_ACDC label1 max {label1.max()} min {label1.min()} label2 max {label2.max()} min {label2.min()}")
        # evaluate_ACDC label1 max 3.0 min 0.0 label2 max 3.0 min 0.0
        image1 = F.interpolate(image1.float(), size=[192, 224, 160], mode='trilinear')
        image2 = F.interpolate(image2.float(), size=[192, 224, 160], mode='trilinear')
        # print(f"evaluate_ACDC image1 {image1.shape} image2 {image2.shape}")
        label1 = F.interpolate(label1.float(), size=[192, 224, 160], mode='trilinear')
        label2 = F.interpolate(label2.float(), size=[192, 224, 160], mode='trilinear')
        # print(f"evaluate_ACDC label1 {label1.shape} label2 {label2.shape}")

        with torch.no_grad():
            start = time.time()
            # _, _, _, agg_flow = model.module(image1, image2)
            _, _, _, agg_flow = model(image1, image2)
            end = time.time()
            times = end - start

        jaccs = []
        dices = []

        # print(f"eval_dataset.seg_values {eval_dataset.seg_values}")
        # eval_dataset.seg_values [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
        # 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        for v in eval_dataset.seg_values:
            label1_fixed = mask_class(label1, v)
            # label2_warped = warp3D()(mask_class(label2, v), agg_flow)
            label2_in = mask_class(label2, v)
            label2_warped = warp3D()(label2_in, agg_flow)
            # print(f"evaluate_ACDC label1_fixed {label1_fixed.shape} label2_warped {label2_warped.shape}")
            # evaluate_ACDC label1_fixed torch.Size([1, 1, 15, 288, 232]) label2_warped torch.Size([1, 1, 15, 288, 232])
            # print(f"evaluate_ACDC label1_fixed max {label1_fixed.max()} min {label1_fixed.min()} label2_in max {label2_in.max()} min {label2_in.min()}")
            # evaluate_OASIS label1_fixed max 255.0 min 0.0 label2_in max 255.0 min 0.0
            # print(f"evaluate_ACDC label2_warped max {label2_warped.max()} min {label2_warped.min()}")
            # evaluate_OASIS label2_warped max 255.00003051757812 min 0.0
            class_dice, class_jacc = mask_metrics(label1_fixed, label2_warped)

            dices.append(class_dice)
            jaccs.append(class_jacc)

        jacb = jacobian_det(agg_flow.cpu().numpy()[0])

        dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
        jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()

        # if args.local_rank == 0:
            # print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
            #         format(i, dice, jacc, jacb, times),
            #         file=f, flush=True)
            # print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
            #         format(i, dice, jacc, jacb, times))
        Logging.info('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                    format(i, dice, jacc, jacb, times))

        Dice.append(dice)
        Jacc.append(jacc)
        Jacb.append(jacb)
        Time.append(times)

    dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
    jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
    jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))
    time_mean, time_std = np.mean(np.array(Time[1:])), np.std(np.array(Time[1:]))

    # if args.local_rank == 0:
        # print('Summary --->  '
        #         'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
        #         'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
        #         .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
        #         file=f, flush=True)
        # print('Summary --->  '
        #         'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
        #         'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
        #         .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))
        # print('Step{:12d} --->  '
        #         'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
        #         'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
        #         .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
        #         file=g, flush=True)
        # print('Step{:12d} --->  '
        #         'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
        #         'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
        #         .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))
        
    Logging.info('Summary --->  '
        'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
        'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
        .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

    Logging.info('Step{:12d} --->  '
            'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
            'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
            .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

    # f.close()
    # g.close()


def train_teacher(args, Logging):
    model = Framework_Teacher(args)
    model.cuda()
    Logging.info(f"RDN model: {model}")

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # if args.local_rank == 0:
    #     count_parameters(model, type="teacher")
    count_parameters(model, Logging, type="teacher")

    # if args.restore_teacher_ckpt is not None:
    #     if args.local_rank == 0:
    #         print('Restore ckpt: %s' % args.restore_ckpt, file=args.files, flush=True)
    #         print('Restore ckpt: %s' % args.restore_ckpt)
    #     model.load_state_dict(torch.load(args.restore_ckpt))

    model.train()
    model.cuda()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    # logger = Logger(model, scheduler, args)
    logger = Logger(model, scheduler, Logging, args)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            model.train()
            # image1-fixed  image2-moving
            image1, image2 = [x.cuda() for x in data_blob]
            # print(f"train_teacher image1 {image1.shape} image2 {image2.shape}")
            # # train_teacher image1 torch.Size([1, 1, 9, 216, 256]) image2 torch.Size([1, 1, 9, 216, 256])
            image1 = F.interpolate(image1.float(), size=[192, 224, 160], mode='trilinear')
            image2 = F.interpolate(image2.float(), size=[192, 224, 160], mode='trilinear')
            # print(f"image1 {image1.shape} image2 {image2.shape}")
            # image1 torch.Size([1, 1, 192, 224, 160]) image2 torch.Size([1, 1, 192, 224, 160])

            optimizer.zero_grad()
            image2_aug = augmentation(image2)
            # Logging.info(f"RDN training image1: {image1.shape} image2_aug: {image2_aug.shape}")
            # RDN training image1: torch.Size([1, 1, 192, 224, 160]) image2_aug: torch.Size([1, 1, 192, 224, 160])

            image2_aug, _, deforms, agg_flow = model(image1, image2_aug)

            loss, metrics = fetch_loss(_, deforms, agg_flow, image1, image2_aug)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps = total_steps + 1
            # print(f"total_steps: {total_steps}")
            logger.push(metrics)

            # if total_steps % args.val_freq == args.val_freq - 1:
            if total_steps % args.val_freq == 0:
                # if args.local_rank == 0:
                model.eval()
                evaluate_ACDC(args, model, total_steps + 1, Logging, type="teacher")
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                torch.save(model.state_dict(), PATH)

            if total_steps == args.num_steps:
                should_keep_training = False
                break

    PATH = args.model_path + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACDC', help='which dataset to use for training')
    parser.add_argument('--restore_teacher_ckpt', type=str, default=None, help='restore and train from this teacher checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--round', type=int, default=5000, help='number of batches per epoch')
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=10) # 50
    parser.add_argument('--val_freq', type=int, default=5000) # 250
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    # args.task_path = sys.path[0]
    # if args.task_path[:5] == "/code":
    #     args.task_path = "/"
    #     args.name = str(args.task_path[5:])
    # else:
    #     args.name = os.path.split(args.task_path)[1]
    args.name = "RDN"
    args.base_path = "/mnt/lhz/Github/Image_registration/RDN_checkpoints"
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    # args.model_path = args.task_path + '/output/checkpoints_' + args.dataset
    args.model_path = os.path.join(args.base_path, args.dataset + '_' + curr_time, 'checkpoints')
    # args.eval_path = args.task_path + '/output/eval_' + args.dataset
    args.eval_path = os.path.join(args.base_path, args.dataset + '_' + curr_time, 'eval')

    os.makedirs(args.base_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)

    # if args.local_rank == 0:
    #     make_dirs(args)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    args.nums_gpu = torch.cuda.device_count()
    args.batch = args.batch
    args.num_steps = args.epoch * args.round

    if args.dataset == "ACDC" or "OASIS":
        # args.dataset_val = ['ACDC_val']
        args.data_path = "/mnt/lhz/Github/Image_registration/RDN/images/" 
    else:
        print('Wrong Dataset')

    # Logger
    Logging = logging.getLogger()
    Logging.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.model_path, 'train_' + args.dataset + '.log'))
    file_handler.setLevel(logging.INFO)
    Logging.addHandler(file_handler)
    Logging.addHandler(stdout_handler)

    # args.files = open(args.task_path + '/output/train_' + args.dataset + '.txt', 'a+')
    # if args.local_rank == 0:
        # print('Dataset: %s' % args.dataset, file=args.files, flush=True)
        # print('Batch size: %s' % args.batch, file=args.files, flush=True)
        # print('Step: %s' % args.num_steps, file=args.files, flush=True)
        # print("Path: %s" % args.task_path, file=args.files, flush=True)
        # print('Parallel GPU: %s' % args.nums_gpu, file=args.files, flush=True)

        # print('Dataset: %s' % args.dataset)
        # print('Batch size: %s' % args.batch)
        # print('Step: %s' % args.num_steps)
        # print("Path: %s" % args.task_path)
        # print('Parallel GPU: %s' % args.nums_gpu)

    Logging.info('Dataset: %s' % args.dataset)
    Logging.info('Batch size: %s' % args.batch)
    Logging.info('Step: %s' % args.num_steps)
    Logging.info('Path: %s' % args.base_path)
    Logging.info('Parallel GPU: %s' % args.nums_gpu)
    Logging.info(f"now cuda device: {torch.cuda.current_device()}")

    train_teacher(args, Logging)
    # args.files.close()

    print("Finished!")