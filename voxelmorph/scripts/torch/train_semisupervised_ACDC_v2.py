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

def register(model, epoch, args):
    # load moving and fixed images
    add_feat_axis = not args.multichannel
    
    inshape = (192, 192, 192)
    pairlist = [f.split(' ') for f in read_files_txt(args.test_txt_path)]
    
    Dice = HD95 = 0
    # model.eval()
    with torch.no_grad():
        for p in pairlist:
            moving_img, moving_seg, fixed_img, fixed_seg = p[0], p[1], p[2], p[3]
            warped_img = moving_img.split('/')[-1].split('_')[0] + '_img_warped_ep'+ str(epoch) +'.nii.gz'
            warped_seg = moving_img.split('/')[-1].split('_')[0] + '_seg_warped_ep'+ str(epoch) +'.nii.gz'
            warped_flow = moving_img.split('/')[-1].split('_')[0] + '_flow_warped_ep'+ str(epoch) +'.nii.gz'
            print(f"moving_img: {moving_img} moving_seg: {moving_seg} fixed_img: {fixed_img} fixed_seg: {fixed_seg}")
            print(f"warped_img: {warped_img} warped_seg: {warped_seg}")
            # moving_img: /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient107/patient107_frame10.nii.gz moving_seg: /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient107/patient107_frame10_gt.nii.gz fixed_img: /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient107/patient107_frame01.nii.gz fixed_seg: /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient107/patient107_frame01_gt.nii.gz
            # warped_img: patient107_img_warped_ep0_.nii.gz warped_seg: patient107_seg_warped_ep0_.nii.gz
            
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
            # semisupervised_pairs labels: [0 1 2 3]
            src_seg = split_seg(moving_seg, labels)
            trg_seg = split_seg(fixed_seg, labels)
            
            print(f"register moving: {moving.shape} moving_seg: {moving_seg.shape}")
            print(f"register fixed: {fixed.shape} fixed_seg: {fixed_seg.shape}")
            print(f"register src_seg: {src_seg.shape} trg_seg: {trg_seg.shape}")
            # register moving: (1, 256, 216, 9, 1) moving_seg: (1, 256, 216, 9, 1)
            # register fixed: (1, 256, 216, 9, 1) fixed_seg: (1, 256, 216, 9, 1)
            # register src_seg: (1, 256, 216, 9, 4) trg_seg: (1, 256, 216, 9, 4)
            
            input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
            input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
            input_moving_seg = torch.from_numpy(src_seg).to(device).float().permute(0, 4, 1, 2, 3)
            input_fixed_seg = torch.from_numpy(trg_seg).to(device).float().permute(0, 4, 1, 2, 3)
            input_moving = F.interpolate(input_moving, size=inshape)
            input_fixed = F.interpolate(input_fixed, size=inshape)
            input_moving_seg = F.interpolate(input_moving_seg, size=inshape)
            input_fixed_seg = F.interpolate(input_fixed_seg, size=inshape)
            
            print(f"register input_moving: {input_moving.shape} input_fixed: {input_fixed.shape} input_moving_seg: {input_moving_seg.shape}")
            
            # moved_img, warp, moved_seg = model(input_moving, input_fixed, input_moving_seg, registration=True)
            moved_img, warp, moved_seg = model(input_moving, input_fixed, input_moving_seg)
            
            data_in = sitk.ReadImage(p[2])
            shape_img = data_in.GetSize()
            ED_origin = data_in.GetOrigin()
            ED_direction = data_in.GetDirection()
            ED_spacing = data_in.GetSpacing()
            # print(f"register fixed image shape: {shape_img}")
            # register fixed image shape: (256, 216, 10)
            
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
            
            
    return Dice, HD95


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
                                                        atlas_file=args.atlas)

    # extract shape from sampled input
    # inshape = next(generator)[0][0].shape[1:-1]
    # print(f"next(generator)[0][0]: {next(generator)[0][0].shape}")
    # next(generator)[0][0]: (1, 216, 256, 8, 1)
    # inshape = next(generator)[0][0].shape[1:-1]
    # inshape = (160, 192, 224)
    inshape = (192, 192, 192)

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
        model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=inshape,
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

    losses += [vxm.losses.Dice().loss]
    weights += [args.weight]
    
    best_dice = best_HD95 = 0

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
            inputs = [F.interpolate(d, size=inshape) for d in inputs]
            y_true = [F.interpolate(d, size=inshape) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                # print(f"n: {n}  y_true: {y_true[n].shape} y_pred: {y_pred[n].shape}")
                # n: 0  y_true: torch.Size([1, 1, 160, 192, 224]) y_pred: torch.Size([1, 1, 160, 192, 224])
                # n: 1  y_true: torch.Size([1, 3, 160, 192, 224]) y_pred: torch.Size([1, 3, 80, 96, 112])
                # n: 2  y_true: torch.Size([1, 4, 160, 192, 224]) y_pred: torch.Size([1, 4, 160, 192, 224])
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
                
            print(f"Training epoch: {epoch} -- step: {step} loss: {', '.join(['%.4e' % f for f in loss_list])}")

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        logger.info(f"{epoch_info} - {time_info} - {loss_info}")
        
        # save model checkpoint
        if epoch % 1 == 0:
            Dice, HD95 = register(model, epoch, args)
            model.save(os.path.join(args.checkpoint_dir, '%04d.pt' % epoch))
            logger.info(f"Saving model to: {os.path.join(args.checkpoint_dir, '%04d.pt' % epoch)}")


    # final model save
    # model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
    model.save(os.path.join(args.checkpoint_dir, 'final.pt'))


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
    parser.add_argument('--int-downsize', type=int, default=1,
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
    args.checkpoint_dir = os.path.join(model_dir, "VoxelMorph_ACDC_seg_" + curr_time)
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
    
    
    
    


