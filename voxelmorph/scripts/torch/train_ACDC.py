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

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep83


def train(args, logger, device, checkpoint_dir):
    bidir = args.bidir

    # load and prepare training data
    # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/py/utils.py
    # train_files = vxm.py.utils.read_file_list(args.img_list, 
    #                                           prefix=args.img_prefix,
    #                                           suffix=args.img_suffix)
    train_files = vxm.py.utils.read_pair_list(args.img_list,
                                            prefix=args.img_prefix,
                                            suffix=args.img_suffix)
    assert len(train_files) > 0, 'Could not find any training data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel

    # print(f"*** atlas: {args.atlas}")
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
        # generator = vxm.generators.scan_to_scan(train_files, 
        #                                         batch_size=args.batch_size,
        #                                         bidir=args.bidir, 
        #                                         add_feat_axis=add_feat_axis)
        generator = vxm.generators.scan_to_scan_pairs(train_files, 
                                                batch_size=args.batch_size,
                                                bidir=args.bidir, 
                                                add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    # inshape = next(generator)[0][0].shape[1:-1]
    # print(f"next(generator)[0][0]: {next(generator)[0][0].shape}")
    # next(generator)[0][0]: (1, 216, 256, 8, 1)
    # inshape = next(generator)[0][0].shape[1:-1]
    inshape = (160, 192, 224)

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
        model = vxm.networks.VxmDense(inshape=inshape,
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
    model.train()
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

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            model.save(os.path.join(checkpoint_dir, '%04d.pt' % epoch))
            logger.info(f"Saving model to: {os.path.join(checkpoint_dir, '%04d.pt' % epoch)}")

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(args.steps_per_epoch):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            inputs = [F.interpolate(d, size=(160, 192, 224)) for d in inputs]
            y_true = [F.interpolate(d, size=(160, 192, 224)) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            # print(f"inputs[0]: {inputs[0].shape} inputs[1]: {inputs[1].shape} n input: {len(inputs)} ")
            # print(f"y_true[0]: {y_true[0].shape} y_true[1]: {y_true[1].shape} n y_true: {len(y_true)} ")
            # inputs: torch.Size([1, 1, 216, 256, 9]) y_true: torch.Size([1, 1, 216, 256, 9])
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

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
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        logger.info(f"{epoch_info} - {time_info} - {loss_info}")

    # final model save
    # model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
    model.save(os.path.join(checkpoint_dir, 'final.pt'))


if __name__ == "__main__":

    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

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
    checkpoint_dir = os.path.join(model_dir, "VoxelMorph_"+curr_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(checkpoint_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.info(f"Config: {args}")

    train(args, logger, device, checkpoint_dir)
    
    
    
    


