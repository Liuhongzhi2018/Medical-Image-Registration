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
    elif args.dataset == 'ACDC' or args.dataset == 'OASIS' or args.dataset == 'OAIZIB':
        train_dataset = datasets.ACDCTrain(args)
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


class Logger:
    def __init__(self, model, scheduler, Logging, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.sum_freq = args.sum_freq

    def _print_training_status(self):
        metrics_data = ["{" + k + ":{:10.5f}".format(self.running_loss[k] / self.sum_freq) + "} "
                        for k in self.running_loss.keys()]
        training_str = "[Steps:{:9d}, Lr:{:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_lr()[0])
        # print(training_str + "".join(metrics_data), file=args.files, flush=True)
        # print(training_str + "".join(metrics_data))
        Logging.info(training_str + "".join(metrics_data))

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


def jacobian_determinant(disp):
    disp = np.expand_dims(disp.transpose(3, 2, 1, 0), 0)
    # print(f"jacobian_determinant disp: {disp.shape}")
    # jacobian_determinant disp: (1, 3, 160, 192, 160)
    _, _, H, W, D = disp.shape
    disp_one = disp[0][0]
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    # return jacdet
    nonpjacdet = np.sum(jacdet <= 0)/np.prod(disp_one.shape)
    return nonpjacdet

# https://docs.monai.io/en/stable/metrics.html
# https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
def Dice(pred, gt):
    # intersection = np.logical_and(gt, pred)
    # union = np.logical_or(gt, pred)
    # dice = 2 * (intersection + smooth)/(mask_sum + smooth)
    smooth = 0.001
    intersection = np.logical_and(pred, gt)
    mask_sum =  np.sum(np.abs(pred)) + np.sum(np.abs(gt))
    # dice = 2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
    dice = 2.0 * (np.sum(intersection) + smooth) / (mask_sum + smooth)
    return dice

# https://github.com/OldaKodym/evaluation_metrics/blob/master/metrics.py
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

# https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
def IOU(pred, gt, classes=1):
    '''
    Intersection over Union (IoU) and Jaccard coefficients (or indices)
    The Jaccard index is also known as Intersection over Union (IoU)
    '''
    smooth = 0.001
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou

# https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/ecfd6f368e9c8e64417120ddbe06562054757a89/L2R_LUMIR_Eval/utils.py
def calc_TRE(dfm_lms, fx_lms, spacing_mov=1):
    '''
    Target Registration Error (TRE)
    '''
    x = np.linspace(0, fx_lms.shape[0] - 1, fx_lms.shape[0])
    y = np.linspace(0, fx_lms.shape[1] - 1, fx_lms.shape[1])
    z = np.linspace(0, fx_lms.shape[2] - 1, fx_lms.shape[2])
    yv, xv, zv = np.meshgrid(y, x, z)
    unique = np.unique(fx_lms)
    smooth = 0.001
    dfm_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (dfm_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + smooth)
        yc = np.sum(label * yv) / (np.sum(label) + smooth)
        zc = np.sum(label * zv) / (np.sum(label) + smooth)
        dfm_pos[i - 1, 0] = xc
        dfm_pos[i - 1, 1] = yc
        dfm_pos[i - 1, 2] = zc

    fx_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (fx_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + smooth)
        yc = np.sum(label * yv) / (np.sum(label) + smooth)
        zc = np.sum(label * zv) / (np.sum(label) + smooth)
        fx_pos[i - 1, 0] = xc
        fx_pos[i - 1, 1] = yc
        fx_pos[i - 1, 2] = zc

    dfm_fx_error = np.mean(np.sqrt(np.sum(np.power((dfm_pos - fx_pos)*spacing_mov, 2), 1)))
    # print(('landmark error (vox): after {}'.format(dfm_fx_error)))
    return dfm_fx_error


def compute_per_class_Dice_HD95_IOU_TRE_NDV(pre, gt, gtspacing):
    n_dice_list = []
    n_hd95_list = []
    n_iou_list = []
    class_num = np.unique(gt)
    pred_num = np.unique(pre)
    # print(f"compute_per_class_Dice_HD95_IOU_TRE_NDV: {class_num} {pred_num}")

    for c in class_num:
        if c == 0: continue
        # print(f"class_num {c}")
        ngt_data = np.zeros_like(gt)
        ngt_data[gt == c] = 1
        npred_data = np.zeros_like(pre)
        npred_data[pre == c] = 1
        npred_data = 1 - ngt_data if 0 == np.count_nonzero(npred_data) else npred_data
        # n_dice = 2*np.sum(ngt_data*npred_data)/(np.sum(1*ngt_data+npred_data) + 0.0001)
        n_dice = Dice(npred_data, ngt_data)
        # print(f"ngt_data: {np.unique(ngt_data)} npred_data: {np.unique(npred_data)}")
        n_hd95 = hd95(npred_data, ngt_data, voxelspacing = gtspacing[::-1])
        n_iou = IOU(npred_data, ngt_data)
        n_dice_list.append(n_dice)
        n_hd95_list.append(n_hd95)
        n_iou_list.append(n_iou)
    mean_Dice = statistics.mean(n_dice_list)
    mean_HD95 = statistics.mean(n_hd95_list)
    mean_iou = statistics.mean(n_iou_list)
    
    tre = calc_TRE(ngt_data, npred_data)
    return tre, mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list


def evaluate(args, Logging, eval_dataset, img_size, model, total_steps):
    Dice, Jacc, Jacb = [], [], []
    mdice_list, mhd95_list, mIOU_list, tre_list, jd_list = [], [], [], [], []
    for i in range(len(eval_dataset)):
        fpath = eval_dataset[i][0]
        mov, fixed = eval_dataset[i][1][np.newaxis].cuda(), eval_dataset[i][2][np.newaxis].cuda()
        mov_label, fixed_label = eval_dataset[i][3][np.newaxis].cuda(), eval_dataset[i][4][np.newaxis].cuda()
        # print(f"evaluate mov shape: {mov.shape} fixed shape: {fixed.shape}")
        # print(f"evaluate mov_label shape: {mov_label.shape} fixed_label shape: {fixed_label.shape}")
        # evaluate mov shape: torch.Size([1, 1, 232, 288, 15]) fixed shape: torch.Size([1, 1, 232, 288, 15])
        # evaluate mov_label shape: torch.Size([1, 1, 232, 288, 15]) fixed_label shape: torch.Size([1, 1, 232, 288, 15])
        # print(f"evaluate mov_label: {torch.unique(mov_label)}")
        # print(f"evaluate fixed_label: {torch.unique(fixed_label)}")

        image1 = F.interpolate(mov, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        image2 = F.interpolate(fixed, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        label1 = F.interpolate(mov_label, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        label2 = F.interpolate(fixed_label, size=img_size, mode='trilinear').permute(0, 1, 4, 3, 2)
        # print(f"transpose mov shape: {image1.shape} fixed shape: {image2.shape}")
        # print(f"transpose mov_label shape: {label1.shape} fixed_label shape: {label2.shape}")
        # transpose mov shape: torch.Size([1, 1, 80, 80, 80]) fixed shape: torch.Size([1, 1, 80, 80, 80])
        # transpose mov_label shape: torch.Size([1, 1, 80, 80, 80]) fixed_label shape: torch.Size([1, 1, 80, 80, 80])
        label1, label2 = torch.round(label1), torch.round(label2)
        # print(f"mov label: {torch.unique(label1)}")
        # print(f"fixed label: {torch.unique(label2)}")
        # mov label: tensor([0., 1., 2., 3.], device='cuda:0')
        # fixed label: tensor([0., 1., 2., 3.], device='cuda:0')

        with torch.no_grad():
            # _, _, _, agg_flow, _ = model.module(image1, image2, augment=False)
            _, _, _, agg_flow, _ = model(image2, image1, augment=False)

            # print(f"eval_dataset seg_values: {eval_dataset.seg_values}")
            # eval_dataset seg_values: [0 1 2 3]
            image1_warped = warp3D()(image1, agg_flow)
            # print(f"transpose mov shape: {image1.shape} fixed shape: {image2.shape}")
            # transpose mov shape: torch.Size([1, 1, 80, 80, 80]) fixed shape: torch.Size([1, 1, 80, 80, 80])
            label1_warped = warp3D()(label1, agg_flow)
            # print(f"warp_seg: {torch.unique(label1_warped)}")
            # warp_seg: tensor([0.0000e+00, 1.1788e-05, 7.0557e-05,  ..., 3.0000e+00, 3.0000e+00, 3.0000e+00], device='cuda:0')

            name = fpath.split('/')[-1][:10]
            # print(f"register mov_path: {mov_path}")
            data_in = sitk.ReadImage(fpath)
            shape_img = data_in.GetSize()
            # print(f"shape_img: {shape_img}")  # (Width, Height, Depth)
            # shape_img: (232, 288, 15)
            ED_origin = data_in.GetOrigin()
            ED_direction = data_in.GetDirection()
            ED_spacing = data_in.GetSpacing()

            # warp_img = F.interpolate(image1_warped, size=shape_img, mode='trilinear')
            warp_img = F.interpolate(image1_warped, size=(shape_img[2], shape_img[1], shape_img[0]), mode='trilinear')
            # warp_img_array = warp_img.detach().cpu().numpy().squeeze().transpose(2, 1, 0)
            warp_img_array = warp_img.detach().cpu().numpy().squeeze()
            
            # warp_seg = F.interpolate(label1_warped, size=shape_img, mode='trilinear')
            warp_seg = F.interpolate(label1_warped, size=(shape_img[2], shape_img[1], shape_img[0]), mode='trilinear')
            # warp_seg_array = warp_seg.detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(np.uint8)
            warp_seg_array = warp_seg.detach().cpu().numpy().squeeze().astype(np.uint8)
            # print(f"warp_seg label: {np.unique(warp_seg_array)}")
            # warp_seg label: [0 1 2 3]

            label2_seg = F.interpolate(label2, size=(shape_img[2], shape_img[1], shape_img[0]), mode='trilinear')
            gt_seg_array = label2_seg.detach().cpu().numpy().squeeze().astype(np.uint8)
            
            tre, mdice, mhd95, mIOU, dice_list, hd95_list, IOU_list = compute_per_class_Dice_HD95_IOU_TRE_NDV(warp_seg_array, gt_seg_array, ED_spacing)

            savedSample_warped = sitk.GetImageFromArray(warp_img_array)
            savedSample_seg = sitk.GetImageFromArray(warp_seg_array)
            # savedSample_defm = sitk.GetImageFromArray(deform)
            
            savedSample_warped.SetOrigin(ED_origin)
            savedSample_seg.SetOrigin(ED_origin)
            # savedSample_defm.SetOrigin(ED_origin)
            
            savedSample_warped.SetDirection(ED_direction)
            savedSample_seg.SetDirection(ED_direction)
            # savedSample_defm.SetDirection(ED_direction)
            
            savedSample_warped.SetSpacing(ED_spacing)
            savedSample_seg.SetSpacing(ED_spacing)
            # savedSample_defm.SetSpacing(ED_spacing)
            
            warped_img_path = os.path.join(args.eval_path, name + '_step' + str(total_steps) + '_warped_img.nii.gz')
            warped_seg_path = os.path.join(args.eval_path, name + '_step' + str(total_steps) + '_warped_seg.nii.gz')
            # warped_flow_path = os.path.join(sample_dir, name + '_ep' + str(epoch) + '_warped_deformflow.nii.gz')
            
            sitk.WriteImage(savedSample_warped, warped_img_path)
            sitk.WriteImage(savedSample_seg, warped_seg_path)
            # sitk.WriteImage(savedSample_defm, warped_flow_path)

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
            
            # Logging.info(f'Total_steps: {total_steps} Pair: {i:6d}/{len(eval_dataset):6d} dice:{dice:10.6f} jacc:{jacc:10.6f} jacb:{jacb:10.2f}')
            Logging.info(f'Total steps: {total_steps} Pair: {i:d}/{len(eval_dataset):d} dice:{dice:.6f} jacc:{jacc:.6f} jacb:{jacb:.6f}')
            
            Dice.append(dice)
            Jacc.append(jacc)
            Jacb.append(jacb)

            Logging.info(f"Total steps: {total_steps} {name} mean Dice {mdice} - {', '.join(['%.4e' % f for f in dice_list])}")
            Logging.info(f"Total steps: {total_steps} {name} mean HD95 {mhd95} - {', '.join(['%.4e' % f for f in hd95_list])}")
            Logging.info(f"Total steps: {total_steps} {name} mean IOU {mIOU} - {', '.join(['%.4e' % f for f in IOU_list])}")
            # Logging.info(f"Total steps: {total_steps} {name} jacobian_determinant - {jd}")
            
            mdice_list.append(mdice)
            mhd95_list.append(mhd95)
            mIOU_list.append(mIOU)
            tre_list.append(tre)
            # jd_list.append(jd)

    dice_mean, dice_std = np.mean(np.array(Dice)), np.std(np.array(Dice))
    jacc_mean, jacc_std = np.mean(np.array(Jacc)), np.std(np.array(Jacc))
    jacb_mean, jacb_std = np.mean(np.array(Jacb)), np.std(np.array(Jacb))

    # Logging.info(f'Total_steps {total_steps:12d} ---> \
    #             Dice:{dice_mean:10.6f}({dice_std:10.6f}) \
    #             Jacc:{jacc_mean:10.6f}({jacc_std:10.6f}) \
    #             Jacb:{jacb_mean:10.2f}({jacb_std:10.2f})')
    Logging.info(f'Total_steps {total_steps:d} ---> \
                Dice:{dice_mean:.6f}({dice_std:.6f}) \
                Jacc:{jacc_mean:.6f}({jacc_std:.6f}) \
                Jacb:{jacb_mean:.6f}({jacb_std:.6f})')

    Logging.info(f"mdice_list {mdice_list} mhd95_list {mhd95_list} mIOU_list {mIOU_list} tre_list {tre_list}")

    cur_avg_dice, cur_avg_hd95, cur_avg_iou = np.mean(mdice_list), np.mean(mhd95_list), np.mean(mIOU_list)
    cur_meanTre = np.mean(tre_list)
    # cur_meanjd = np.mean(jd_list)
    
    Logging.info(f"Total steps: {total_steps} - avgDice: {cur_avg_dice} avgHD95: {cur_avg_hd95} avgIOU: {cur_avg_iou} avgTRE: {cur_meanTre}")    
    # Epoch: 0 - avgDice: 0.3953122517969288 avgHD95: 10.5551533471546 avgIOU: 0.2547768406802452 avgTRE: 5.765068821433447 avgJD: 0.0



def train(args, Logging):

    model = Framework(args)
    model.cuda()
    Logging.info(f"SDHNet model: {model}")
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader = fetch_dataloader(args, Logging)
    # img_size = (160, 192, 224)
    # img_size = (80, 80, 80)
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

            # if total_steps % args.val_freq == args.val_freq - 1:
            if total_steps % args.val_freq == 0:
                # PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps + 1)
                PATH = args.model_path + '/%s_%d.pth' % (args.name, total_steps)
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
    parser.add_argument('--dataset', type=str, default='OAIZIB', help='which dataset to use for training')
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
