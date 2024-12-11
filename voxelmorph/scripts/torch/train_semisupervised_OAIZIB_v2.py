#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import sys
import random
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import time
import logging
import SimpleITK as sitk
import statistics
import scipy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, \
                                     binary_erosion,\
                                     generate_binary_structure

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep83

def read_files_txt(txt_path):
    with open(txt_path, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content]
    return filelist

# internal utility to generate downsampled prob seg from discrete seg
def split_seg(seg, labels):
    prob_seg = np.zeros((*seg.shape[:4], len(labels)))
    for i, label in enumerate(labels):
        prob_seg[0, ..., i] = seg[0, ..., 0] == label
    # return prob_seg[:, ::downsize, ::downsize, ::downsize, :]
    return prob_seg

def jacobian_determinant(disp):
    disp = np.expand_dims(disp.transpose(3, 2, 1, 0), 0)
    # print(f"jacobian_determinant disp: {disp.shape}")
    # jacobian_determinant disp: (1, 3, 214, 256, 9)
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

def iou(pred, gt, classes=1):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

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

def calc_TRE(dfm_lms, fx_lms, spacing_mov=1):
    x = np.linspace(0, fx_lms.shape[0] - 1, fx_lms.shape[0])
    y = np.linspace(0, fx_lms.shape[1] - 1, fx_lms.shape[1])
    z = np.linspace(0, fx_lms.shape[2] - 1, fx_lms.shape[2])
    yv, xv, zv = np.meshgrid(y, x, z)
    unique = np.unique(fx_lms)

    dfm_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (dfm_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + 1e-3)
        yc = np.sum(label * yv) / (np.sum(label) + 1e-3)
        zc = np.sum(label * zv) / (np.sum(label) + 1e-3)
        dfm_pos[i - 1, 0] = xc
        dfm_pos[i - 1, 1] = yc
        dfm_pos[i - 1, 2] = zc

    fx_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (fx_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + 1e-3)
        yc = np.sum(label * yv) / (np.sum(label) + 1e-3)
        zc = np.sum(label * zv) / (np.sum(label) + 1e-3)
        fx_pos[i - 1, 0] = xc
        fx_pos[i - 1, 1] = yc
        fx_pos[i - 1, 2] = zc

    dfm_fx_error = np.mean(np.sqrt(np.sum(np.power((dfm_pos - fx_pos)*spacing_mov, 2), 1)))
    return dfm_fx_error

# /mnt/lhz/Github/Image_registration/voxelmorph/./scripts/torch/train_semisupervised_LPBA.py:127: RuntimeWarning: invalid value encountered in scalar divide
#   xc = np.sum(label * xv) / np.sum(label)
# /mnt/lhz/Github/Image_registration/voxelmorph/./scripts/torch/train_semisupervised_LPBA.py:128: RuntimeWarning: invalid value encountered in scalar divide
#   yc = np.sum(label * yv) / np.sum(label)
# /mnt/lhz/Github/Image_registration/voxelmorph/./scripts/torch/train_semisupervised_LPBA.py:129: RuntimeWarning: invalid value encountered in scalar divide
#   zc = np.sum(label * zv) / np.sum(label)


def compute_per_class_Dice_HD95_IOU_TRE_NDV(pre, gt, gtspacing):
    n_dice_list = []
    n_hd95_list = []
    n_iou_list = []
    class_num = np.unique(gt)
    for c in class_num:
        if c == 0: continue
        ngt_data = np.zeros_like(gt)
        ngt_data[gt == c] = 1
        npred_data = np.zeros_like(pre)
        npred_data[pre == c] = 1
        if np.count_nonzero(npred_data) == 0: npred_data = np.ones_like(pre)
        # n_dice = 2*np.sum(ngt_data*npred_data)/(np.sum(1*ngt_data+npred_data) + 0.0001)
        n_dice = Dice(npred_data, ngt_data)
        n_hd95 = hd95(npred_data, ngt_data, voxelspacing = gtspacing[::-1])
        # n_iou = iou(npred_data, ngt_data)
        n_iou = IOU(npred_data, ngt_data)
        n_dice_list.append(n_dice)
        n_hd95_list.append(n_hd95)
        n_iou_list.append(n_iou)
    mean_Dice = statistics.mean(n_dice_list)
    mean_HD95 = statistics.mean(n_hd95_list)
    mean_iou = statistics.mean(n_iou_list)
    
    # tre = calc_TRE(ngt_data, npred_data)
    tre = calc_TRE(gt, pre)
    return tre, mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list

def register(model, epoch, logger, args):
    # load moving and fixed images
    add_feat_axis = not args.multichannel
    
    # inshape = (160, 384, 384)
    # inshape = (160, 160, 160)
    inshape = (192, 192, 192)
    pairlist = [f.split(' ') for f in read_files_txt(args.test_txt_path)]
    mdice_list, mhd95_list, mIOU_list, tre_list, jd_list = [], [], [], [], []
    # model.eval()
    # with torch.no_grad():
    for p in pairlist:
        moving_img, moving_seg, fixed_img, fixed_seg = p[0], p[1], p[2], p[3]
        warped_img = moving_img.split('/')[-1].split('.')[0] + '_' + fixed_img.split('/')[-1].split('.')[0] + '_ep' + str(epoch) + '_warped_img' +'.nii.gz'
        warped_seg = moving_img.split('/')[-1].split('.')[0] + '_' + fixed_img.split('/')[-1].split('.')[0] + '_ep' + str(epoch) + '_warped_seg' +'.nii.gz'
        warped_flow = moving_img.split('/')[-1].split('.')[0] + '_' + fixed_img.split('/')[-1].split('.')[0] + '_ep' + str(epoch) + '_warped_deformflow' +'.nii.gz'
      
        moving = vxm.py.utils.load_volfile(moving_img,
                                            np_var='vol',
                                            add_batch_axis=True, 
                                            add_feat_axis=add_feat_axis)
        
        moving_seg = vxm.py.utils.load_volfile(moving_seg,
                                                np_var='seg',
                                                add_batch_axis=True, 
                                                add_feat_axis=add_feat_axis)
        
        fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_img,
                                                        np_var='vol',
                                                        add_batch_axis=True, 
                                                        add_feat_axis=add_feat_axis, 
                                                        ret_affine=True)
        
        fixed_seg = vxm.py.utils.load_volfile(fixed_seg,
                                                np_var='seg',
                                                add_batch_axis=True, 
                                                add_feat_axis=add_feat_axis)

        labels = np.unique(fixed_seg)
        # print(f"semisupervised_pairs labels: {labels}")
        # semisupervised_pairs labels: [0 1 2 3 4 5]
        src_seg = split_seg(moving_seg, labels)
        trg_seg = split_seg(fixed_seg, labels)
        # print(f"semisupervised_pairs src_seg: {src_seg.shape} trg_seg: {trg_seg.shape}")
        # semisupervised_pairs src_seg: (1, 160, 384, 384, 6) trg_seg: (1, 160, 384, 384, 6)
                
        input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
        input_moving_seg = torch.from_numpy(src_seg).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed_seg = torch.from_numpy(trg_seg).to(device).float().permute(0, 4, 1, 2, 3)
        input_moving = F.interpolate(input_moving, size=inshape)
        input_fixed = F.interpolate(input_fixed, size=inshape)
        input_moving_seg = F.interpolate(input_moving_seg, size=inshape)
        input_fixed_seg = F.interpolate(input_fixed_seg, size=inshape)
        # print(f"register input_moving: {input_moving.shape} ")
        # print(f"input_fixed: {input_fixed.shape}")
        # print(f"input_moving_seg: {input_moving_seg.shape}")
        
        # moved_img, warp, moved_seg = model(input_moving, input_fixed, input_moving_seg, registration=True)
        moved_img, warp, moved_seg = model(input_moving, input_fixed, input_moving_seg)
        
        data_in = sitk.ReadImage(p[2])
        shape_img = data_in.GetSize()
        ED_origin = data_in.GetOrigin()
        ED_direction = data_in.GetDirection()
        ED_spacing = data_in.GetSpacing()
        # print(f"register fixed image shape: {shape_img}")
        # register fixed image shape: (256, 216, 10)
        data_seg = sitk.ReadImage(p[3])
        fixed_seg_array = sitk.GetArrayFromImage(data_seg)
        
        moved_img = F.interpolate(moved_img, size=shape_img)
        moved_img = moved_img.detach().cpu().numpy().squeeze().transpose(2, 1, 0)
        # print(f"register out moved_img: {moved_img.shape}")
        # register out moved_img: (192, 192, 192)
        # register out moved_img: (10, 216, 256)
        img_path = os.path.join(args.sample_dir, warped_img)
        # vxm.py.utils.save_volfile(moved_img, img_path, fixed_affine)
        
        # moved_seg = moved_seg.detach().cpu().numpy().squeeze()
        moved_seg = F.interpolate(moved_seg, size=shape_img)
        moved_seg = torch.argmax(moved_seg.squeeze(), dim=0).detach().cpu().numpy().transpose(2, 1, 0).astype(np.uint8)
        # print(f"register out moved_seg: {moved_seg.shape}")
        # register out moved_seg: (4, 192, 192, 192)
        # register out moved_seg: (10, 216, 256)
        seg_path = os.path.join(args.sample_dir, warped_seg)
        # vxm.py.utils.save_volfile(moved_seg, seg_path, fixed_affine)

        warp = F.interpolate(warp, size=shape_img)
        deform = warp.detach().cpu().numpy().squeeze().transpose(3, 2, 1, 0)
        # print(f"register out warp: {deform.shape}")
        # register out warp: (3, 192, 192, 192)
        # register out warp: (10, 216, 256, 3)
        warp_path = os.path.join(args.sample_dir, warped_flow)
        # vxm.py.utils.save_volfile(warp, warp_path, fixed_affine)
        
        # print(f"------ Saving training samples epoch: {epoch} {moving_img.split('/')[-1]} ------")
        savedSample_warped = sitk.GetImageFromArray(moved_img)
        savedSample_seg = sitk.GetImageFromArray(moved_seg)
        savedSample_defm = sitk.GetImageFromArray(deform)
        
        savedSample_warped.SetOrigin(ED_origin)
        savedSample_seg.SetOrigin(ED_origin)
        savedSample_defm.SetOrigin(ED_origin)
        
        savedSample_warped.SetDirection(ED_direction)
        savedSample_seg.SetDirection(ED_direction)
        savedSample_defm.SetDirection(ED_direction)
        
        savedSample_warped.SetSpacing(ED_spacing)
        savedSample_seg.SetSpacing(ED_spacing)
        savedSample_defm.SetSpacing(ED_spacing)
        
        sitk.WriteImage(savedSample_warped, img_path)
        sitk.WriteImage(savedSample_seg, seg_path)
        sitk.WriteImage(savedSample_defm, warp_path)
        
        # print(f"moved_seg: {moved_seg.shape} fixed_seg_array: {fixed_seg_array.shape}")
        tre, mdice, mhd95, mIOU, dice_list, hd95_list, IOU_list = compute_per_class_Dice_HD95_IOU_TRE_NDV(moved_seg, fixed_seg_array, ED_spacing)
        jd = jacobian_determinant(deform)
        logger.info(f"Epoch: {epoch} {moving_img.split('/')[-1]} mean Dice {mdice} - {', '.join(['%.4e' % f for f in dice_list])}")
        logger.info(f"Epoch: {epoch} {moving_img.split('/')[-1]} mean HD95 {mhd95} - {', '.join(['%.4e' % f for f in hd95_list])}")
        logger.info(f"Epoch: {epoch} {moving_img.split('/')[-1]} mean IOU {mIOU} - {', '.join(['%.4e' % f for f in IOU_list])}")
        logger.info(f"Epoch: {epoch} {moving_img.split('/')[-1]} jacobian_determinant - {jd}")
        
        mdice_list.append(mdice)
        mhd95_list.append(mhd95)
        mIOU_list.append(mIOU)
        tre_list.append(tre)
        jd_list.append(jd)
        
    print(f"mdice_list {mdice_list} mhd95_list {mhd95_list} mIOU_list {mIOU_list} tre_list {tre_list}")
    cur_avgdice, cur_avghd95, cur_avgiou = np.mean(mdice_list), np.mean(mhd95_list), np.mean(mIOU_list)
    cur_meanTre = np.mean(tre_list)
    cur_meanjd = np.mean(jd_list)
    
    logger.info(f"Epoch: {epoch} - avgDice: {cur_avgdice} avgHD95: {cur_avghd95} avgIOU: {cur_avgiou} avgTRE: {cur_meanTre} avgJD: {cur_meanjd}")
    
    return cur_avgdice, cur_avghd95, cur_avgiou, cur_meanTre, cur_meanjd

def train(args, logger, device):
    bidir = args.bidir

    # load and prepare training data
    # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/py/utils.py
    train_imgs = vxm.py.utils.read_pair_list(args.img_list,
                                             prefix=args.img_prefix,
                                             suffix=args.img_suffix)
    train_segs = vxm.py.utils.read_pair_list(args.seg_list,
                                             prefix=args.seg_prefix,
                                             suffix=args.seg_suffix)
    assert len(train_imgs) > 0, 'Could not find any training data.'

    # parser.add_argument('--labels', required=True, help='label list (npy format) to use in dice loss')
    # load labels file
    # train_labels = np.load(args.labels)

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel

    if args.atlas:
        # scan-to-atlas generator
        # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/py/utils.py
        atlas = vxm.py.utils.load_volfile(args.atlas, 
                                        np_var='vol',
                                        add_batch_axis=True, 
                                        add_feat_axis=add_feat_axis)
        # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/generators.py
        generator = vxm.generators.scan_to_atlas(train_files, 
                                                atlas,
                                                batch_size=args.batch_size, 
                                                bidir=args.bidir,
                                                add_feat_axis=add_feat_axis)
    else:
        # scan-to-scan generator
        # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/generators.py
        generator = vxm.generators.semisupervised_pairs(train_imgs,
                                                        train_segs,
                                                        use_label=False,
                                                        atlas_file=args.atlas)

    # extract shape from sampled input
    # inshape = next(generator)[0][0].shape[1:-1]
    # gen_shape = next(generator)[0][0].shape[1:-1]
    # print(f"next(generator)[0][0]: {gen_shape}")
    # next(generator)[0][0]: (160, 384, 384)
    # gen_shape = (160, 160, 160)
    gen_shape = (192, 192, 192)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, 
                                           device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=gen_shape,
                                                       nb_unet_features=[enc_nf, dec_nf],
                                                       bidir=bidir,
                                                       int_steps=args.int_steps,
                                                       int_downsize=args.int_downsize)

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    # model.train()
    logger.info(f"Model: {model}")

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]

    # # prepare Dice loss
    # losses += [vxm.losses.Dice().loss]
    # weights += [args.weight]
    
    best_epoch, best_avg_Dice, best_avg_HD95, best_avg_iou, best_avg_tre = 0, 0, 10000, 0, 10000
    # training loops
    for epoch in range(args.initial_epoch, args.epochs):
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        model.train()
        for step in range(args.steps_per_epoch):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            inputs = [F.interpolate(d, size=gen_shape) for d in inputs]
            y_true = [F.interpolate(d, size=gen_shape) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                # print(f"n: {n}  y_true: {y_true[n].shape} y_pred: {y_pred[n].shape}")
                # n: 0  y_true: torch.Size([1, 1, 160, 224, 192]) y_pred: torch.Size([1, 1, 160, 224, 192])
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            # print(f"Training epoch: {epoch} -- step: {step} loss: {', '.join(['%.4e' % f for f in loss_list])}")
            logger.info(f"Training epoch: {epoch} -- step: {step}/{args.steps_per_epoch} loss: {', '.join(['%.4e' % f for f in loss_list])}")
            
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        # epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        epoch_info = 'Epoch %d/%d' % (epoch, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        logger.info(f"{epoch_info} - {time_info} - {loss_info}")
        
        # save model checkpoint
        if epoch % 500 == 0:
            with torch.no_grad():
                cur_avg_dice, cur_avg_hd95, cur_avg_iou, cur_meanTre, cur_meanjd = register(model, epoch, logger, args)
                    
            # if cur_avg_dice > best_avg_Dice and cur_avg_hd95 < best_avg_HD95 and cur_avg_iou > best_avg_iou and cur_meanTre < best_avg_tre:
            if cur_avg_dice > best_avg_Dice and cur_avg_hd95 < best_avg_HD95 and cur_avg_iou > best_avg_iou:
                best_epoch = epoch
                best_avg_Dice = cur_avg_dice
                best_avg_HD95 = cur_avg_hd95
                best_avg_iou = cur_avg_iou
                # best_avg_tre = cur_meanTre
                model.save(os.path.join(args.checkpoint_dir, 'best_model.pth'))
                # print(f"Saving best model to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
                logger.info(f"Saving best model to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
                
        for f in os.listdir(args.sample_dir):
            if "ep" + str(best_epoch) in f: continue
            else:
                os.remove(os.path.join(args.sample_dir, f))
                # print(f"remove samples without < epoch {best_epoch} >")
                
        for f in os.listdir(args.checkpoint_dir):
            if '%04d.pth' % best_epoch in f or 'log' in f or 'best' in f: continue
            if not os.path.isdir(os.path.join(args.checkpoint_dir, f)):
                os.remove(os.path.join(args.checkpoint_dir, f))
            # print(f"remove pth without < epoch {best_epoch} >")
            
        model.save(os.path.join(args.checkpoint_dir, '%04d.pth' % epoch))
        logger.info(f"Saving model to: {os.path.join(args.checkpoint_dir, '%04d.pth' % epoch)}")

        print(f"Epoch: {epoch} Current Dice {cur_avg_dice} HD95 {cur_avg_hd95} IOU {cur_avg_iou} TRE {cur_meanTre} nonpJD {cur_meanjd} Best_Dice {best_avg_Dice} Best_HD95 {best_avg_HD95} Best_IOU {best_avg_iou} at epoch {best_epoch}")
        logger.info(f"Epoch: {epoch} Current Dice {cur_avg_dice} HD95 {cur_avg_hd95} IOU {cur_avg_iou} TRE {cur_meanTre} nonpJD {cur_meanjd} Best_Dice {best_avg_Dice} Best_HD95 {best_avg_HD95} Best_IOU {best_avg_iou} at epoch {best_epoch}")
  
    # final model save
    # model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
    model.save(os.path.join(args.checkpoint_dir, 'final.pth'))
    

if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--seg-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--seg-prefix', help='optional input segment file prefix')
    parser.add_argument('--seg-suffix', help='optional input segment file suffix')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    parser.add_argument('--test-txt-path', required=True, help='moving image (source) filename')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    # parse configs
    args = parser.parse_args()
    print(f"Config: {args}")

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.checkpoint_dir = os.path.join(model_dir, "VoxelMorph_semi_OAIZIB_" + curr_time)
    args.sample_dir = os.path.join(args.checkpoint_dir, "samples")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.checkpoint_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.info(f"Config: {args}")

    train(args, logger, device)
