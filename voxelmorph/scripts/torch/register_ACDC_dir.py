#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8


def read_files_txt(txt_path):
    with open(txt_path, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content]
    return filelist

def register(moving_file, fixed_file, moved_file, warp_file, new_moving_file, new_fixed_file, args):
    # load moving and fixed images
    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(moving_file, 
                                       add_batch_axis=True, 
                                       add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_file, 
                                                    add_batch_axis=True, 
                                                    add_feat_axis=add_feat_axis, 
                                                    ret_affine=True)

    # load and set up model
    model = vxm.networks.VxmDense.load(args.model, device)
    model.to(device)
    model.eval()
    
    # set up tensors and permute
    orishape = moving.shape[1:-1]
    inshape = (160, 192, 224)
    print(f"moving: {moving.shape} fixed: {fixed.shape} orishape: {orishape}")
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
    input_moving = F.interpolate(input_moving, size=inshape)
    input_fixed = F.interpolate(input_fixed, size=inshape)
    print(f"input_moving: {input_moving.shape} input_fixed: {input_fixed.shape}")
    # input_moving: torch.Size([1, 1, 160, 192, 224]) input_fixed: torch.Size([1, 1, 160, 192, 224])
    
    # predict
    moved, warp = model(input_moving, input_fixed, registration=True)
    print(f"moved: {moved.shape} warp: {warp.shape}")
    # moved: torch.Size([1, 1, 160, 192, 224]) warp: torch.Size([1, 3, 160, 192, 224])

    # save moved image
    # if args.moved:
    # moved = moved.detach().cpu().numpy().squeeze()
    # # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/py/utils.py
    # vxm.py.utils.save_volfile(moved, moved_file, fixed_affine)
    moved = F.interpolate(moved, size=orishape).detach().cpu().numpy().squeeze()
    # /home/liuhongzhi/Method/Registration/voxelmorph/voxelmorph/py/utils.py
    vxm.py.utils.save_volfile(moved, moved_file, fixed_affine)

    # save warp
    # if args.warp:
    # warp = warp.detach().cpu().numpy().squeeze()
    # vxm.py.utils.save_volfile(warp, warp_file, fixed_affine)
    warp = F.interpolate(warp, size=orishape).detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, warp_file, fixed_affine)
    
    input_moving_save = F.interpolate(input_moving, size=orishape).detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(input_moving_save, new_moving_file, fixed_affine)
    input_fixed_save = F.interpolate(input_fixed, size=orishape).detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(input_fixed_save, new_fixed_file, fixed_affine)


if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--moving', required=True, help='moving image (source) filename')
    # parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    # parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--txt_path', required=True, help='moving image (source) filename')
    parser.add_argument('--moved_dir', required=True, help='warped image output filename')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('--warp', help='output warp deformation filename')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    args = parser.parse_args()

    # device handling
    if args.gpu and (args.gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    os.makedirs(args.moved_dir, exist_ok=True)
        
    pairlist = [f.split(' ') for f in read_files_txt(args.txt_path)]
    # print(f"pairlist: {pairlist}")
    count = 0
    for p in pairlist:
        count += 1
        moving_file, fixed_file = p[0], p[1]    # ED, ES
        # moving_file, fixed_file = p[1], p[0]    # ES, ED
        moved_file = os.path.join(args.moved_dir, moving_file.split('/')[-1].split('.')[0] + '_moved.nii.gz')
        warp_file = os.path.join(args.moved_dir, moving_file.split('/')[-1].split('.')[0] + '_warp.nii.gz')
        # print(f"* moving_file: {moving_file} fixed_file: {fixed_file}")
        # print(f"** moved_file: {moved_file} warp_file: {warp_file}")
        new_moving_file = os.path.join(args.moved_dir, moving_file.split('/')[-1].split('.')[0] + '_moving.nii.gz')
        new_fixed_file = os.path.join(args.moved_dir, moving_file.split('/')[-1].split('.')[0] + '_fixed.nii.gz')
        register(moving_file, fixed_file, moved_file, warp_file, new_moving_file, new_fixed_file, args)
        print(f"count: {count+1} {moving_file.split('/')[-1]}")
    # register()