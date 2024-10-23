import os
import argparse
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
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

def count_parameters(model, type):
    print("\n----------" + type.title() + "-Net Params----------", file=args.files, flush=True)
    print("\n----------" + type.title() + "-Net Params----------")
    print('Whole parameters: %d'
          % sum(p.numel() for p in model.parameters() if p.requires_grad),
          file=args.files, flush=True)
    print('Whole parameters: %d'
          % sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Affine parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad),
          file=args.files, flush=True)
    print('Affine parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad))
    print('Deform parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad),
          file=args.files, flush=True)
    print('Deform parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad))
    print("----------" + type.title() + "-Net Params----------\n", file=args.files, flush=True)
    print("----------" + type.title() + "-Net Params----------\n")


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
    if args.dataset == 'liver':
        train_dataset = datasets.LiverTrain(args)
    elif args.dataset == 'brain':
        train_dataset = datasets.BrainTrain(args)
    elif args.dataset == 'oasis':
        train_dataset = datasets.OasisTrain(args)
    else:
        print('Wrong Dataset')

    gpuargs = {'num_workers': 4, 'drop_last': True}
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, pin_memory=True, shuffle=False,
                              sampler=train_sampler, **gpuargs)

    if args.local_rank == 0:
        print('Image pairs in training: %d' % len(train_dataset), file=args.files, flush=True)
        print('Image pairs in training: %d' % len(train_dataset))
    return train_loader


def fetch_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    milestones = [args.round * 3, args.round * 4, args.round * 5]  # args.epoch == 5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    return optimizer, scheduler


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


def train_teacher(args):
    model = Framework_Teacher(args)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if args.local_rank == 0:
        count_parameters(model, type="teacher")

    if args.restore_teacher_ckpt is not None:
        if args.local_rank == 0:
            print('Restore ckpt: %s' % args.restore_ckpt, file=args.files, flush=True)
            print('Restore ckpt: %s' % args.restore_ckpt)
        model.load_state_dict(torch.load(args.restore_ckpt))

    model.train()
    model.cuda()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, args)

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            model.train()
            image1, image2 = [x.cuda() for x in data_blob]

            optimizer.zero_grad()
            image2_aug = augmentation(image2)
            image2_aug, _, deforms, agg_flow = model(image1, image2_aug)

            loss, metrics = fetch_loss(_, deforms, agg_flow, image1, image2_aug)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps = total_steps + 1

            logger.push(metrics)

            if total_steps % args.val_freq == args.val_freq - 1:
                if args.local_rank == 0:
                    model.eval()
                    evaluate(args, model, total_steps + 1, type="teacher")
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                torch.save(model.state_dict(), PATH)

            if total_steps == args.num_steps:
                should_keep_training = False
                break

    PATH = args.model_path + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oasis', help='which dataset to use for training')
    parser.add_argument('--restore_teacher_ckpt', type=str, default=None, help='restore and train from this teacher checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--round', type=int, default=5000, help='number of batches per epoch')
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=50)
    parser.add_argument('--val_freq', type=int, default=250)

    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    args.task_path = sys.path[0]
    if args.task_path[:5] == "/code":
        args.task_path = "/"
        args.name = str(args.task_path[5:])
    else:
        args.name = os.path.split(args.task_path)[1]

    args.model_path = args.task_path + '/output/checkpoints_' + args.dataset
    args.eval_path = args.task_path + '/output/eval_' + args.dataset

    if args.local_rank == 0:
        make_dirs(args)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    args.nums_gpu = torch.cuda.device_count()
    args.batch = args.batch
    args.num_steps = args.epoch * args.round


    if args.dataset == "liver":
        args.dataset_val = ['sliver_val', 'lspig_val']
        if args.task_path == "/":
            args.data_path = "/data/hubosist/CT_Liver_tiff/"
        else:
            args.data_path = "/braindat/lab/hubo/DATASET/CT_Liver/"
    elif args.dataset == "brain":
        args.dataset_val = ["lpba_val"]
        if args.task_path == "/":
            args.data_path = "/data/hubosist/MRI_Brain_tiff/"
        else:
            args.data_path = "/braindat/lab/hubo/DATASET/MRI_Brain/"
    elif args.dataset == "oasis":
        args.dataset_val = ["oasis_val"]
        if args.task_path == "/":
            args.data_path = "/data/hubosist/MRI_Brain_tiff/"
        else:
            args.data_path = "/braindat/lab/hubo/DATASET/Learn2Reg/OASIS/"
    else:
        print('Wrong Dataset')


    args.files = open(args.task_path + '/output/train_' + args.dataset + '.txt', 'a+')
    if args.local_rank == 0:
        print('Dataset: %s' % args.dataset, file=args.files, flush=True)
        print('Batch size: %s' % args.batch, file=args.files, flush=True)
        print('Step: %s' % args.num_steps, file=args.files, flush=True)
        print("Path: %s" % args.task_path, file=args.files, flush=True)
        print('Parallel GPU: %s' % args.nums_gpu, file=args.files, flush=True)

        print('Dataset: %s' % args.dataset)
        print('Batch size: %s' % args.batch)
        print('Step: %s' % args.num_steps)
        print("Path: %s" % args.task_path)
        print('Parallel GPU: %s' % args.nums_gpu)

    train_teacher(args)
    args.files.close()

    print("Finished!")