import os
import numpy as np
from os.path import join
import argparse
import time
import sys
import torch
import torch.nn as nn
import re
import core.datasets as datasets
from core.utils.warp import warp3D
# from core.framework import Framework_Teacher


def count_parameters(model, f):
    print('Whole parameters: %d'
          % sum(p.numel() for p in model.parameters() if p.requires_grad),
          file=f, flush=True)
    print('Whole parameters: %d'
          % sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Affine parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad),
          file=f, flush=True)
    print('Affine parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'affnet' in name and p.requires_grad))
    print('Deform parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad),
          file=f, flush=True)
    print('Deform parameters: %d'
          % sum(p.numel() for name, p in model.named_parameters() if 'defnet' in name and p.requires_grad))


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


def evaluate_liver(args, model, steps, type):
    for datas in args.dataset_val:
        eval_path = join(args.eval_path, type, datas)
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)
        file_sum = join(eval_path, datas + '.txt')
        file = join(eval_path, datas + '_' + str(steps) + '.txt')
        f = open(file, 'a+')
        g = open(file_sum, 'a+')

        Dice, Jacc, Jacb, Time = [], [], [], []
        if 'lspig' in datas:
            eval_dataset = datasets.LspigTest(args, datas)
        else:
            eval_dataset = datasets.LiverTest(args, datas)
        if args.local_rank == 0:
            print('Dataset in evaluation: %s' % datas, file=f, flush=True)
            print('Dataset in evaluation: %s' % datas)
            print('Image pairs in evaluation: %d' % len(eval_dataset), file=f, flush=True)
            print('Image pairs in evaluation: %d' % len(eval_dataset))
            print('Evaluation steps: %s' % steps, file=f, flush=True)
            print('Evaluation steps: %s' % steps)
            print('Model Type: %s' % type, file=f, flush=True)
            print('Model Type: %s' % type)
        count_parameters(model, f)

        for i in range(len(eval_dataset)):
            image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
            label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()

            with torch.no_grad():
                start = time.time()
                _, _, _, agg_flow = model.module(image1, image2)
                end = time.time()
                times = end - start
                label2_warped = warp3D()(label2, agg_flow)

            dice, jacc = mask_metrics(label1, label2_warped)

            dice = dice.cpu().numpy()[0]
            jacc = jacc.cpu().numpy()[0]
            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            if args.local_rank == 0:
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times),
                      file=f, flush=True)
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times))

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)
            Time.append(times)

        dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
        jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
        jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))
        time_mean, time_std = np.mean(np.array(Time[1:])), np.std(np.array(Time[1:]))

        if args.local_rank == 0:
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=f, flush=True)
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=g, flush=True)
            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))


        f.close()
        g.close()


def evaluate_oasis(args, model, steps, type):
    for datas in args.dataset_val:
        eval_path = join(args.eval_path, type, datas)
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)
        file_sum = join(eval_path, datas + '.txt')
        file = join(eval_path, datas + '_' + str(steps) + '.txt')
        f = open(file, 'a+')
        g = open(file_sum, 'a+')

        Dice, Jacc, Jacb, Time = [], [], [], []
        eval_dataset = datasets.OasisTest(args)
        if args.local_rank == 0:
            print('Dataset in evaluation: %s' % datas, file=f, flush=True)
            print('Dataset in evaluation: %s' % datas)
            print('Image pairs in evaluation: %d' % len(eval_dataset), file=f, flush=True)
            print('Image pairs in evaluation: %d' % len(eval_dataset))
            print('Evaluation steps: %s' % steps, file=f, flush=True)
            print('Evaluation steps: %s' % steps)
            print('Model Type: %s' % type, file=f, flush=True)
            print('Model Type: %s' % type)
        count_parameters(model, f)

        for i in range(len(eval_dataset)):
            image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
            label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()

            with torch.no_grad():
                start = time.time()
                _, _, _, agg_flow = model.module(image1, image2)
                end = time.time()
                times = end - start

            jaccs = []
            dices = []

            for v in eval_dataset.seg_values:
                label1_fixed = mask_class(label1, v)
                label2_warped = warp3D()(mask_class(label2, v), agg_flow)

                class_dice, class_jacc = mask_metrics(label1_fixed, label2_warped)

                dices.append(class_dice)
                jaccs.append(class_jacc)

            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
            jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()

            if args.local_rank == 0:
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times),
                      file=f, flush=True)
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times))

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)
            Time.append(times)

        dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
        jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
        jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))
        time_mean, time_std = np.mean(np.array(Time[1:])), np.std(np.array(Time[1:]))

        if args.local_rank == 0:
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=f, flush=True)
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=g, flush=True)
            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

        f.close()
        g.close()

def evaluate_brain(args, model, steps, type):
    for datas in args.dataset_val:
        eval_path = join(args.eval_path, type, datas)
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)
        file_sum = join(eval_path, datas + '.txt')
        file = join(eval_path, datas + '_' + str(steps) + '.txt')
        f = open(file, 'a+')
        g = open(file_sum, 'a+')

        Dice, Jacc, Jacb, Time = [], [], [], []
        eval_dataset = datasets.BrainTest(args, datas)
        if args.local_rank == 0:
            print('Dataset in evaluation: %s' % datas, file=f, flush=True)
            print('Dataset in evaluation: %s' % datas)
            print('Image pairs in evaluation: %d' % len(eval_dataset), file=f, flush=True)
            print('Image pairs in evaluation: %d' % len(eval_dataset))
            print('Evaluation steps: %s' % steps, file=f, flush=True)
            print('Evaluation steps: %s' % steps)
            print('Model Type: %s' % type, file=f, flush=True)
            print('Model Type: %s' % type)
        count_parameters(model, f)

        for i in range(len(eval_dataset)):
            image1, image2 = eval_dataset[i][0][np.newaxis].cuda(), eval_dataset[i][1][np.newaxis].cuda()
            label1, label2 = eval_dataset[i][2][np.newaxis].cuda(), eval_dataset[i][3][np.newaxis].cuda()

            with torch.no_grad():
                start = time.time()
                _, _, _, agg_flow = model.module(image1, image2)
                end = time.time()
                times = end - start

            jaccs = []
            dices = []

            for v in eval_dataset.seg_values:
                label1_fixed = mask_class(label1, v)
                label2_warped = warp3D()(mask_class(label2, v), agg_flow)

                class_dice, class_jacc = mask_metrics(label1_fixed, label2_warped)

                dices.append(class_dice)
                jaccs.append(class_jacc)

            jacb = jacobian_det(agg_flow.cpu().numpy()[0])

            dice = torch.mean(torch.cuda.FloatTensor(dices)).cpu().numpy()
            jacc = torch.mean(torch.cuda.FloatTensor(jaccs)).cpu().numpy()

            if args.local_rank == 0:
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times),
                      file=f, flush=True)
                print('Pair{:6d}   dice:{:10.6f}   jacc:{:10.6f}   new_jacb:{:10.2f}   time:{:10.6f}'.
                      format(i, dice, jacc, jacb, times))

            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)
            Time.append(times)

        dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
        jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
        jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))
        time_mean, time_std = np.mean(np.array(Time[1:])), np.std(np.array(Time[1:]))

        if args.local_rank == 0:
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=f, flush=True)
            print('Summary --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std),
                  file=g, flush=True)
            print('Step{:12d} --->  '
                  'Dice:{:10.6f}({:10.6f})   Jacc:{:10.6f}({:10.6f})  '
                  'New_Jacb:{:10.2f}({:10.2f})   Time:{:10.6f}({:10.6f})'
                  .format(steps, dice_mean, dice_std, jacc_mean, jacc_std, jacb_mean, jacb_std, time_mean, time_std))

        f.close()
        g.close()


def evaluate(args, model, steps, type):
    if args.dataset == 'liver':
        evaluate_liver(args, model, steps, type)
    elif args.dataset == 'brain':
        evaluate_brain(args, model, steps, type)
    elif args.dataset == 'oasis':
        evaluate_oasis(args, model, steps, type)
    else:
        print('Wrong Dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='liver', help='which dataset to use for evaluation')
    parser.add_argument('--dataset_val', nargs='+', default=['sliver_val'], help='specific dataset to use for evaluation')
    parser.add_argument('--restore_ckpt', type=str, default="/braindat/lab/hubo/CODE/Registration/HB/Exp_3rd/Scripts/check1/VM_distill_liver_no_share_weight_loss3/ckpt/VM_distill_liver_no_share_weight_loss3_92000.pth", help="the ckpt path")
    parser.add_argument('--data_path', type=str, default='/braindat/lab/hubo/DATASET/CT_Liver/')
    parser.add_argument('--base_path', type=str, default='/braindat/lab/hubo/CODE/Registration/HB/Exp_3rd/Scripts/')

    args = parser.parse_args()

    args.task_path = sys.path[0]
    args.eval_path = args.task_path + '/output/eval_' + args.dataset

    args.name = os.path.split(args.task_path)[1]

    model = Framework_Teacher(args)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()

    model.load_state_dict(torch.load(args.restore_ckpt))

    args.restore_step = re.findall("(?<=_)[1-9][0-9]*(?=\.)", args.restore_ckpt)[0]
    evaluate(args, model, int(args.restore_step), type="student")
