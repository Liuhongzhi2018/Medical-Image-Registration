# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# python imports
import os
import glob
import json
import time
import tempfile
import warnings
from pprint import pprint
import argparse
import logging
import shutil
import sys
import random
from ignite.contrib.handlers import ProgressBar

# data science imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
# RuntimeError: Pin memory thread exited unexpectedly
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# MONAI imports
import monai
from monai.apps import download_and_extract
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data.utils import list_data_collate
from monai.losses import BendingEnergyLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import SegResNet
from monai.utils import set_determinism, first
from monai.config import print_config
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.transforms import (
    Compose,
    MapTransform,
    Resized,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


# CUDA_VISIBLE_DEVICES=2 python Ankle_segment_trainval_net.py train --data_folder "../data/Task01_177_3label" --model_folder "../segment_checkpoints/ankleseg_12156_3label_UNet"
# CUDA_VISIBLE_DEVICES=1 python Ankle_registration_trainval_net.py train --data_folder "./data/Task12_121_55" --model_folder "../regist_checkpoints/"


def get_files(data_dir):
    """
    Get L2R train/val files from NLST challenge
    """
    data_json = os.path.join(data_dir, "NLST_dataset.json")

    with open(data_json) as file:
        data = json.load(file)

    train_files = []
    for pair in data["training_paired_images"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        train_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    val_files = []
    for pair in data["registration_val"]:
        nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
        nam_moving = os.path.basename(pair["moving"]).split(".")[0]
        val_files.append(
            {
                "fixed_image": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
            }
        )

    return train_files, val_files


def get_files_1(data_dir):

    train_files = [
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0004_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0004_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0007_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0007_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0010_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0010_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0013_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0013_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0016_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0016_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0019_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0019_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0022_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0022_0000.nii.gz"
        }
    ]
    
    val_files = [
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0025_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0025_0000.nii.gz"
        },
        {
            "fixed_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_image": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/imagesTr/AbdomenCTCT_0028_0000.nii.gz",
            "fixed_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0001_0000.nii.gz",
            "moving_label": "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT/labelsTr/AbdomenCTCT_0028_0000.nii.gz"
        }
    ]

    return train_files, val_files


def overlay_img(img1, img2, slice_idx, ax, title=None):
    ax.imshow(1 - img1[:, slice_idx, :].T, cmap="Blues", origin="lower")
    ax.imshow(1 - img2[:, slice_idx, :].T, cmap="Oranges", origin="lower", alpha=0.5)
    if title is not None:
        ax.title.set_text(title)
        

def plot_images(target_res, fixed_image, moving_image, fixed_label, moving_label, model_folder):
    # Image and label visualization
    slice_idx = int(target_res[0] * 95.0 / 224)  # at full-res (224 slices), visualize sagittal slice 95
    fig, axs = plt.subplots(2, 3)
    # plot images
    axs[0, 0].imshow(fixed_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[0, 1].imshow(moving_image[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 2])
    # plot labels
    axs[1, 0].imshow(fixed_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    axs[1, 1].imshow(moving_label[:, slice_idx, :].T, cmap="bone", origin="lower")
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 2])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations")
    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(model_folder, "Image_and_label_visualizations.png"))
    print("Saving Image and label visualizations figure")
    plt.close()
    
    
def Pointcloud_visualization(check_data, model_folder):
    # Pointcloud visualization
    fixed_points = check_data["fixed_keypoints"][0]
    moving_points = check_data["moving_keypoints"][0]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2], s=10.0, marker="o", color="lightblue")
    ax.scatter(moving_points[:, 0], moving_points[:, 1], moving_points[:, 2], s=10.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.title("Pointcloud visualizations")
    # plt.show()
    plt.savefig(os.path.join(model_folder, "Pointcloud_visualizations.png"))
    print("Saving Pointcloud visualizations figure")
    plt.close()
    
    
def forward(fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer):
    """
    Model forward pass: predict DDF, warp moving images/labels/keypoints
    """
    batch_size = fixed_image.shape[0]

    # predict DDF through LocalNet
    ddf_image = model(torch.cat((moving_image, fixed_image), dim=1)).float()

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf_image)

    # warp moving label (optional)
    if moving_label is not None:
        pred_label = warp_layer(moving_label, ddf_image)
    else:
        pred_label = None

    # warp vectors for keypoints (optional)
    if fixed_keypoints is not None:
        with torch.no_grad():
            offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
            offset = offset[None][None]
            ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
        ddf_keypoints = (
            F.grid_sample(ddf_image, ddf_keypoints.view(batch_size, -1, 1, 1, 3))
            .view(batch_size, 3, -1)
            .permute((0, 2, 1))
        )
    else:
        ddf_keypoints = None

    return ddf_image, ddf_keypoints, pred_image, pred_label


def forward_image_label(fixed_image, moving_image, moving_label, model, warp_layer):
    """
    Model forward pass: predict DDF, warp moving images/labels/keypoints
    """
    batch_size = fixed_image.shape[0]

    # predict DDF through LocalNet
    ddf_image = model(torch.cat((moving_image, fixed_image), dim=1)).float()

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf_image)

    # warp moving label (optional)
    if moving_label is not None:
        pred_label = warp_layer(moving_label, ddf_image)
    else:
        pred_label = None

    # # warp vectors for keypoints (optional)
    # if fixed_keypoints is not None:
    #     with torch.no_grad():
    #         offset = torch.as_tensor(fixed_image.shape[-3:]).to(fixed_keypoints.device) / 2
    #         offset = offset[None][None]
    #         ddf_keypoints = torch.flip((fixed_keypoints - offset) / offset, (-1,))
    #     ddf_keypoints = (
    #         F.grid_sample(ddf_image, ddf_keypoints.view(batch_size, -1, 1, 1, 3))
    #         .view(batch_size, 3, -1)
    #         .permute((0, 2, 1))
    #     )
    # else:
    #     ddf_keypoints = None
    ddf_keypoints = None

    return ddf_image, ddf_keypoints, pred_image, pred_label


def collate_fn(batch):
    """
    Custom collate function.
    Some background:
        Collation is the "collapsing" of a list of N-dimensional tensors into a single (N+1)-dimensional tensor.
        The `Dataloader` object  performs this step after receiving a batch of (transformed) data from the
        `Dataset` object.
        Note that the `Resized` transform above resamples all image tensors to a shape `spatial_size`,
        thus images can be easily collated.
        Keypoints, however, are of different row-size and thus cannot be easily collated
        (a.k.a. "ragged" or "jagged" tensors): [(n_0, 3), (n_1, 3), ...]
        This function aligns the row-size of these tensors such that they can be collated like
        any regular list of tensors.
        To do this, the max number of keypoints is determined, and shorter keypoint-lists are filled up with NaNs.
        Then, the average-TRE loss below can be computed via `nanmean` aggregation (i.e. ignoring filled-up elements).
    """
    max_length = 0
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        if length > max_length:
            max_length = length
    for data in batch:
        length = data["fixed_keypoints"].shape[0]
        data["fixed_keypoints"] = torch.concat(
            (data["fixed_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )
        data["moving_keypoints"] = torch.concat(
            (data["moving_keypoints"], float("nan") * torch.ones((max_length - length, 3))), dim=0
        )

    return list_data_collate(batch)


def tre(fixed, moving, vx=None):
    """
    Computes target registration error (TRE) loss for keypoint matching.
    """
    if vx is None:
        return ((fixed - moving) ** 2).sum(-1).sqrt().nanmean()
    else:
        return ((fixed - moving).mul(vx) ** 2).sum(-1).sqrt().nanmean()


def loss_fun(
    fixed_image,
    pred_image,
    fixed_label,
    pred_label,
    fixed_keypoints,
    pred_keypoints,
    ddf_image,
    lam_t,
    lam_l,
    lam_m,
    lam_r,
):
    """
    Custom multi-target loss:
        - TRE as main loss component
        - Parametrizable weights for further (optional) components: MSE/BendingEnergy/Dice loss
    Note: Might require "calibration" of lambda weights for the multi-target components,
        e.g. by making a first trial run, and manually setting weights to account for different magnitudes
    """
    # Instantiate where necessary
    if lam_m > 0:
        mse_loss = MSELoss()
    if lam_r > 0:
        regularization = BendingEnergyLoss()
    if lam_l > 0:
        label_loss = DiceLoss()
    # Compute loss components
    t = tre(fixed_keypoints, pred_keypoints) if lam_t > 0 else 0.0
    p = label_loss(pred_label, fixed_label) if lam_l > 0 else 0.0
    m = mse_loss(fixed_image, pred_image) if lam_m > 0 else 0.0
    r = regularization(ddf_image) if lam_r > 0 else 0.0
    # Weighted combination:
    return lam_t * t + lam_l * p + lam_m * m + lam_r * r


def loss_fun_1(
    fixed_image,
    pred_image,
    fixed_label,
    pred_label,
    ddf_image,
    lam_t,
    lam_l,
    lam_m,
    lam_r,
):
    """
    Custom multi-target loss:
        - TRE as main loss component
        - Parametrizable weights for further (optional) components: MSE/BendingEnergy/Dice loss
    Note: Might require "calibration" of lambda weights for the multi-target components,
        e.g. by making a first trial run, and manually setting weights to account for different magnitudes
    """
    # Instantiate where necessary
    if lam_m > 0:
        mse_loss = MSELoss()
    if lam_r > 0:
        regularization = BendingEnergyLoss()
    if lam_l > 0:
        label_loss = DiceLoss()
    # Compute loss components
    # t = tre(fixed_keypoints, pred_keypoints) if lam_t > 0 else 0.0
    p = label_loss(pred_label, fixed_label) if lam_l > 0 else 0.0
    m = mse_loss(fixed_image, pred_image) if lam_m > 0 else 0.0
    r = regularization(ddf_image) if lam_r > 0 else 0.0
    # Weighted combination:
    return lam_l * p + lam_m * m + lam_r * r


def plt_loss(log_train_loss, log_val_dice, log_val_tre):
    # log_val_tre = [x.item() for x in log_val_tre]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(log_train_loss)
    axs[0].title.set_text("train_loss")
    axs[1].plot(log_val_dice)
    axs[1].title.set_text("val_dice")
    axs[2].plot(log_val_tre)
    axs[2].title.set_text("val_tre")
    plt.show()
    
    

def Image_and_label_visualization(check_data, pred_image, pred_label, target_res, fixed_points, moving_points):
    # Image and label visualization
    fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
    fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
    moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
    moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()
    # choose slice
    slice_idx = int(target_res[0] * 95.0 / 224)  # visualize slice 95 at full-res (224 slices)
    # plot images
    fig, axs = plt.subplots(2, 2)
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 0], "Before registration")
    overlay_img(fixed_image, pred_image, slice_idx, axs[0, 1], "After registration")
    # plot labels
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 0])
    overlay_img(fixed_label, pred_label, slice_idx, axs[1, 1])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations pre-/post-registration")
    plt.tight_layout()
    plt.show()

    # Pointcloud visualization
    fixed_keypoints = check_data["fixed_keypoints"][0].cpu()
    moving_keypoints = check_data["moving_keypoints"][0].cpu()
    moved_keypoints = fixed_keypoints + ddf_keypoints[0].cpu()
    # plot pointclouds
    fig = plt.figure()
    # Before registration
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2], s=2.0, marker="o", color="lightblue")
    ax.scatter(moving_points[:, 0], moving_points[:, 1], moving_points[:, 2], s=2.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    # After registration
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(moved_keypoints[:, 0], moved_keypoints[:, 1], moved_keypoints[:, 2], s=2.0, marker="o", color="lightblue")
    ax.scatter(moving_keypoints[:, 0], moving_keypoints[:, 1], moving_keypoints[:, 2], s=2.0, marker="o", color="orange")
    ax.view_init(-10, 80)
    ax.set_aspect("auto")
    plt.show()
    

def Image_and_label_visualization_nop(check_data, pred_image, pred_label, target_res, model_folder):
    # Image and label visualization
    fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
    fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
    moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
    moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()
    # choose slice
    slice_idx = int(target_res[0] * 95.0 / 224)  # visualize slice 95 at full-res (224 slices)
    # plot images
    fig, axs = plt.subplots(2, 2)
    overlay_img(fixed_image, moving_image, slice_idx, axs[0, 0], "Before registration")
    overlay_img(fixed_image, pred_image, slice_idx, axs[0, 1], "After registration")
    # plot labels
    overlay_img(fixed_label, moving_label, slice_idx, axs[1, 0])
    overlay_img(fixed_label, pred_label, slice_idx, axs[1, 1])
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.suptitle("Image and label visualizations pre-/post-registration")
    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(model_folder, "Image_and_label_visualizations_pre-post-registration.png"))
    print("Saving Image and label visualizations pre-/post-registration figure")
    plt.close()

        

class LoadKeypointsd(MapTransform):
    """
    Load keypoints from csv file
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            keypoints = d[key]
            keypoints = pd.read_csv(keypoints, header=None)
            keypoints = keypoints.to_numpy()
            keypoints = torch.as_tensor(keypoints)
            d[key] = keypoints  # [N, 3]
        return d


class TransformKeypointsd(MapTransform):
    """
    Applies any potential linear image transformation to keypoint values
    """

    def __init__(self, keys_keypoints, keys_images, ras=False):
        # super.__init__(self)
        self.keys_keypoints = keys_keypoints
        self.keys_images = keys_images
        self.ras = ras

    def __call__(self, data):
        d = dict(data)
        for kp, ki in zip(self.keys_keypoints, self.keys_images):
            # Get image meta data
            image = d[ki]
            meta = image.meta
            # Get keypoints
            keypoints_ijk = d[kp]
            # Get transformation (in voxel space)
            affine = meta["affine"]
            original_affine = torch.as_tensor(meta["original_affine"], dtype=affine.dtype, device=affine.device)
            transforms_affine = (
                original_affine.inverse() @ affine
            )  # Assumes: affine = original_affine @ transforms_affine
            transforms_affine = transforms_affine.inverse()
            if self.ras:
                # RAS space
                transforms_affine = original_affine @ transforms_affine
            # Apply transformation to keypoints
            keypoints_ijk_moved = torch.cat((keypoints_ijk, torch.ones((keypoints_ijk.shape[0]), 1)), dim=1)
            keypoints_ijk_moved = (transforms_affine @ keypoints_ijk_moved.T).T
            keypoints_ijk_moved = keypoints_ijk_moved[:, :3]
            keypoints_ijk_moved = keypoints_ijk_moved.float()

            d[kp] = keypoints_ijk_moved  # [N, 3]

        return d
    
    
# def infer(model):

#     load_pretrained_model_weights = False
#     if load_pretrained_model_weights:
#         dir_load = dir_save  # folder where network weights are stored
#         # instantiate warp layer
#         warp_layer = Warp().to(device)
#         # instantiate model
#         model = SegResNet(
#             spatial_dims=3,
#             in_channels=2,
#             out_channels=3,
#             blocks_down=[1, 2, 2, 4],
#             blocks_up=[1, 1, 1],
#             init_filters=16,
#             dropout_prob=0.2,
#         )
#         # load model weights
#         filename_best_model = glob.glob(os.path.join(dir_load, "segresnet_kpt_loss_best_tre*"))[0]
#         model.load_state_dict(torch.load(filename_best_model))
#         # to GPU
#         model.to(device)

#     set_determinism(seed=1)
#     check_ds = Dataset(data=val_files, transform=val_transforms)
#     check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
#     check_data = first(check_loader)

#     # Forward pass
#     model.eval()
#     with torch.no_grad():
#         with torch.cuda.amp.autocast(enabled=amp_enabled):
#             ddf_image, ddf_keypoints, pred_image, pred_label = forward(
#                 check_data["fixed_image"].to(device),
#                 check_data["moving_image"].to(device),
#                 check_data["moving_label"].to(device),
#                 check_data["fixed_keypoints"].to(device),
#                 model,
#                 warp_layer,
#             )


def train(data_folder=".", model_folder="runs"):
    """run a training pipeline."""
    
    # train_files, val_files = get_files(data_dir)
    train_files, val_files = get_files_1(data_folder)

    # print 2 training samples to illustrate the contents of the datalist
    pprint(train_files[0:2])
    
    full_res_training = False
    if full_res_training:
        target_res = [224, 192, 224]
        spatial_size = [
            -1,
            -1,
            -1,
        ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
    else:
        target_res = [96, 96, 96]
        spatial_size = target_res  # downsample to 96^3 voxels for faster training on resized data (good for testing)

    # train_transforms = Compose(
    #     [
    #         LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
    #         LoadKeypointsd(
    #             keys=["fixed_keypoints", "moving_keypoints"],
    #         ),
    #         ScaleIntensityRanged(
    #             keys=["fixed_image", "moving_image"],
    #             a_min=-1200,
    #             a_max=400,
    #             b_min=0.0,
    #             b_max=1.0,
    #             clip=True,
    #         ),
    #         Resized(
    #             keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
    #             mode=("trilinear", "trilinear", "nearest", "nearest"),
    #             align_corners=(True, True, None, None),
    #             spatial_size=spatial_size,
    #         ),
    #         RandAffined(
    #             keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
    #             mode=("bilinear", "bilinear", "nearest", "nearest"),
    #             prob=0.8,
    #             shear_range=0.2,
    #             translate_range=int(25 * target_res[0] / 224),
    #             rotate_range=np.pi / 180 * 15,
    #             scale_range=0.2,
    #             padding_mode=("zeros", "zeros", "zeros", "zeros"),
    #         ),
    #         TransformKeypointsd(
    #             keys_keypoints=["fixed_keypoints", "moving_keypoints"],
    #             keys_images=["fixed_image", "moving_image"],
    #         ),
    #     ]
    # )
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-1200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=spatial_size,
            ),
            RandAffined(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                prob=0.8,
                shear_range=0.2,
                translate_range=int(25 * target_res[0] / 224),
                rotate_range=np.pi / 180 * 15,
                scale_range=0.2,
                padding_mode=("zeros", "zeros", "zeros", "zeros"),
            ),
        ]
    )

    # val_transforms = Compose(
    #     [
    #         LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
    #         ScaleIntensityRanged(
    #             keys=["fixed_image", "moving_image"],
    #             a_min=-1200,
    #             a_max=400,
    #             b_min=0.0,
    #             b_max=1.0,
    #             clip=True,
    #         ),
    #         Resized(
    #             keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
    #             mode=("trilinear", "trilinear", "nearest", "nearest"),
    #             align_corners=(True, True, None, None),
    #             spatial_size=spatial_size,
    #         ),
    #         LoadKeypointsd(
    #             keys=["fixed_keypoints", "moving_keypoints"],
    #         ),
    #         TransformKeypointsd(
    #             keys_keypoints=["fixed_keypoints", "moving_keypoints"],
    #             keys_images=["fixed_image", "moving_image"],
    #         ),
    #     ]
    # )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-1200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=("trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(True, True, None, None),
                spatial_size=spatial_size,
            ),
            # LoadKeypointsd(
            #     keys=["fixed_keypoints", "moving_keypoints"],
            # ),
            # TransformKeypointsd(
            #     keys_keypoints=["fixed_keypoints", "moving_keypoints"],
            #     keys_images=["fixed_image", "moving_image"],
            # ),
        ]
    )
    
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)
    
    # Resampled image size
    fixed_image = check_data["fixed_image"][0][0]
    fixed_label = check_data["fixed_label"][0][0]
    moving_image = check_data["moving_image"][0][0]
    moving_label = check_data["moving_label"][0][0]
    print(f"fixed_image shape: {fixed_image.shape}, " f"fixed_label shape: {fixed_label.shape}")
    print(f"moving_image shape: {moving_image.shape}, " f"moving_label shape: {moving_label.shape}")

    # Reorder dims for visualization
    fixed_image = fixed_image.permute(1, 0, 2)
    fixed_label = fixed_label.permute(1, 0, 2)
    moving_image = moving_image.permute(1, 0, 2)
    moving_label = moving_label.permute(1, 0, 2)
    
    plot_images(target_res, fixed_image, moving_image, fixed_label, moving_label, model_folder)
    logging.info(f"Saving Image and label visualizations figure")
    # Pointcloud_visualization(check_data, model_folder)

    # device, optimizer, epoch and batch settings
    device = "cuda:0"
    batch_size = 4
    lr = 1e-4
    weight_decay = 1e-5
    max_epochs = 200

    # image voxel size at target resolution
    vx = np.array([1.5, 1.5, 1.5]) / (np.array(target_res) / np.array([224, 192, 224]))
    vx = torch.tensor(vx).to(device)

    # Use mixed precision feature of GPUs for faster training
    amp_enabled = True

    # loss weights (set to zero to disable loss term)
    lam_t = 1 # 1e0  # TRE  (keypoint loss)
    lam_l = 1 # 0  # Dice (mask overlay)
    lam_m = 1 # 0  # MSE (image similarity)
    lam_r = 1 # 0  # Bending loss (smoothness of the DDF)

    #  Write model and tensorboard logs?
    do_save = True
    dir_save = os.path.join(model_folder, "models", "nlst", "tre-segresnet")
    if do_save and not os.path.exists(dir_save):
        os.makedirs(dir_save)
        
    # Cached datasets for high performance during batch generation
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    # DataLoaders, now with custom function `collate_fn`, to rectify the ragged keypoint tensors
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)    
    # Model
    model = SegResNet(spatial_dims=3,
                      in_channels=2,
                      out_channels=3,
                      blocks_down=[1, 2, 2, 4],
                      blocks_up=[1, 1, 1],
                      init_filters=16,
                      dropout_prob=0.2,).to(device)
    warp_layer = Warp().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Metrics
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    # Automatic mixed precision (AMP) for faster training
    amp_enabled = True
    scaler = torch.cuda.amp.GradScaler()

    # # Tensorboard
    # if do_save:
    #     writer = SummaryWriter(log_dir=dir_save)

    # Start torch training loop
    val_interval = 5
    best_eval_tre = float("inf")
    best_eval_dice = 0
    log_train_loss = []
    log_val_dice = []
    log_val_tre = []
    pth_best_tre, pth_best_dice, pth_latest = "", "", ""
    
    for epoch in range(max_epochs):
        # ==============================================
        # Train
        # ==============================================
        t0_train = time.time()
        model.train()

        epoch_loss, n_steps, tre_before, tre_after = 0, 0, 0, 0
        for batch_data in train_loader:
            # Get data
            fixed_image = batch_data["fixed_image"].to(device)
            moving_image = batch_data["moving_image"].to(device)
            fixed_label = batch_data["fixed_label"].to(device)
            moving_label = batch_data["moving_label"].to(device)
            # fixed_keypoints = batch_data["fixed_keypoints"].to(device)
            # moving_keypoints = batch_data["moving_keypoints"].to(device)
            n_steps += 1
            # Forward pass and loss
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                #     fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
                # )
                ddf_image, ddf_keypoints, pred_image, pred_label = forward_image_label(
                    fixed_image, moving_image, moving_label, model, warp_layer
                )
                loss = loss_fun_1(
                    fixed_image,
                    pred_image,
                    fixed_label,
                    pred_label,
                    ddf_image,
                    lam_t,
                    lam_l,
                    lam_m,
                    lam_r,
                )
            
            # Optimise
            # print(f"loss: {loss}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # # TRE before (voxel space)
            # tre_before += tre(fixed_keypoints, moving_keypoints)
            # tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints)

        # Scheduler step
        lr_scheduler.step()
        # Loss
        epoch_loss /= n_steps
        log_train_loss.append(epoch_loss)
        # if do_save:
        #     writer.add_scalar("train_loss", epoch_loss, epoch)
        print(f"{epoch + 1} | loss={epoch_loss:.6f}")
        logging.info(f"{epoch + 1} | loss={epoch_loss:.6f}")

        # Mean TRE
        # tre_before /= n_steps
        # tre_after /= n_steps
        # print(
        #     (
        #         f"{epoch + 1} | tre_before_train={tre_before:.3f}, tre_after_train={tre_after:.3f}, "
        #         "elapsed time: {time.time()-t0_train:.2f} sec."
        #     )
        # )

        # ==============================================
        # Eval
        # ==============================================
        if (epoch + 1) % val_interval == 0:
            t0_eval = time.time()
            model.eval()

            n_steps, tre_before, tre_after = 0, 0, 0
            with torch.no_grad():
                for batch_data in val_loader:
                    # Get data
                    fixed_image = batch_data["fixed_image"].to(device)
                    moving_image = batch_data["moving_image"].to(device)
                    fixed_label = batch_data["fixed_label"].to(device)
                    moving_label = batch_data["moving_label"].to(device)
                    # fixed_keypoints = batch_data["fixed_keypoints"].to(device)
                    # moving_keypoints = batch_data["moving_keypoints"].to(device)
                    n_steps += 1
                    # Infer
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # ddf_image, ddf_keypoints, pred_image, pred_label = forward(
                        #     fixed_image, moving_image, moving_label, fixed_keypoints, model, warp_layer
                        # )
                        ddf_image, ddf_keypoints, pred_image, pred_label = forward_image_label(
                            fixed_image, moving_image, moving_label, model, warp_layer
                        )
                        ddf_image_check, ddf_keypoints_check, pred_image_check, pred_label_check = forward_image_label(
                            check_data["fixed_image"].to(device),
                            check_data["moving_image"].to(device),
                            check_data["moving_label"].to(device),
                            model,
                            warp_layer,
                        )
                        Image_and_label_visualization_nop(check_data, 
                                                          pred_image_check, 
                                                          pred_label_check, 
                                                          target_res, 
                                                          model_folder)
                        
                    # # TRE
                    # tre_before += tre(fixed_keypoints, moving_keypoints, vx=vx)
                    # tre_after += tre(fixed_keypoints + ddf_keypoints, moving_keypoints, vx=vx)
                    # Dice
                    pred_label = pred_label.round()
                    dice_metric_before(y_pred=moving_label, y=fixed_label)
                    dice_metric_after(y_pred=pred_label, y=fixed_label)
                    
            # Dice
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            # if do_save:
            #     writer.add_scalar("val_dice", dice_after, epoch)
            log_val_dice.append(dice_after)
            print(f"{epoch + 1} | dice_before ={dice_before:.3f}, dice_after ={dice_after:.3f}")
            logging.info(f"{epoch + 1} | dice_before ={dice_before:.3f}, dice_after ={dice_after:.3f}")

            if dice_after > best_eval_dice:
                best_eval_dice = dice_after
                if do_save:
                    # Save best model based on Dice
                    if pth_best_dice != "":
                        os.remove(os.path.join(dir_save, pth_best_dice))
                    pth_best_dice = f"segresnet_kpt_loss_best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                    torch.save(model.state_dict(), os.path.join(dir_save, pth_best_dice))
                    print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")
                    logging.info(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")

        if do_save:
            # Save latest model
            if pth_latest != "":
                os.remove(os.path.join(dir_save, pth_latest))
            pth_latest = "segresnet_kpt_loss_latest.pth"
            torch.save(model.state_dict(), os.path.join(dir_save, pth_latest))
            print(f"Saving latest model")
            logging.info(f"Saving latest model")

if __name__ == "__main__":
    """
    Usage:
        CUDA_VISIBLE_DEVICES=1 python trainval_net.py train --data_folder "ankle_seg_data" --model_folder "checkpoints_0222" # run the training pipeline
        python run_net.py infer --data_folder "ankle_seg_data" --model_folder "checkpoints_0222" # run the inference
        CUDA_LAUNCH_BLOCKING=1 python run_net.py train --data_folder "ankle_seg_data" --model_folder "checkpoints_0222" pipeline
    """
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="runs", type=str, help="model folder")
    args = parser.parse_args()

    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    # model_folder = args.model_folder + "_" + curr_time
    model_folder = os.path.join(args.model_folder, curr_time)
    os.makedirs(model_folder, exist_ok=True)

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    torch.backends.cudnn.benchmark = True
    
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.basicConfig(
        # stream=sys.stdout, 
        level=logging.INFO,
        filename=os.path.join(model_folder, "train.log"),
        filemode='a')
    # random.seed(0)
    
    if args.mode == "train":
        data_folder = args.data_folder or os.path.join("ankle_seg_data", "imagesTr")
        train(data_folder=data_folder, model_folder=model_folder)
    # elif args.mode == "infer":
    #     data_folder = args.data_folder or os.path.join("ankle_seg_data", "imagesTs")
    #     infer(data_folder=data_folder, model_folder=model_folder, prediction_folder=os.path.join(model_folder,"predict"))
    else:
        raise ValueError("Unknown mode.")


# MONAI version: 1.3.0+34.g8e134b8c.dirty
# Numpy version: 1.26.2
# Pytorch version: 1.12.0
# MONAI flags: HAS_EXT = False, USE_COMPILED = False, USE_META_DICT = False
# MONAI rev id: 8e134b8cb92e3c624b23d4d10c5d4596bb5b9d9b
# MONAI __file__: /home/<username>/Github/MONAI/monai/__init__.py

# Optional dependencies:
# Pytorch Ignite version: 0.4.11
# ITK version: 5.3.0
# Nibabel version: 5.1.0
# scikit-image version: 0.22.0
# scipy version: 1.11.4
# Pillow version: 10.0.1
# Tensorboard version: 2.15.1
# gdown version: 4.7.1
# TorchVision version: 0.13.0
# tqdm version: 4.66.1
# lmdb version: 1.4.1
# psutil version: 5.9.6
# pandas version: 2.1.3
# einops version: 0.7.0
# transformers version: 4.21.3
# mlflow version: 2.8.1
# pynrrd version: 1.0.0
# clearml version: 1.13.2
