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

import os
import argparse
import glob
import logging
import shutil
import sys
import random
import time


import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

# RuntimeError: Pin memory thread exited unexpectedly
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.transforms import (
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
# CUDA_VISIBLE_DEVICES=2 python Ankle_segment_trainval_net_v1.py infer --data_folder "./data/Task12_121_55" --model_folder "./segment_checkpoints/ankleseg_Task12_121_55_UNETR_ep500_0403"


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        Orientationd(keys, axcodes="LPS"),
        # Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        # Spacingd(keys, pixdim=(0.20, 0.20, 0.40), mode=("bilinear", "nearest")[: len(keys)]),
        Spacingd(keys, pixdim=(0.45, 0.45, 0.5), mode=("bilinear", "nearest")[: len(keys)]),
        # ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    print("mode: ", mode)
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                # # This is for SwinUNETR
                # SpatialPadd(keys, spatial_size=(128, 128, 32), mode="reflect"),
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                # # This is for SwinUNETR
                # RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(128, 128, 32), num_samples=3), 
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (torch.float32, torch.uint8)
    if mode == "val":
        dtype = (torch.float32, torch.uint8)
    if mode == "infer":
        dtype = (torch.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)


# def get_net():
#     """returns a unet model instance."""

#     # num_classes = 2
#     num_classes = 3
#     net = monai.networks.nets.BasicUNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=num_classes,
#         features=(32, 32, 64, 128, 256, 32),
#         dropout=0.1,
#     )
#     return net


def get_net(num_classes=3):
    """returns a unet model instance."""

    # num_classes = 2
    # num_classes = 3

    # net = monai.networks.nets.BasicUNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     features=(32, 32, 64, 128, 256, 32),
    #     dropout=0.1,
    # )

    # net = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(4, 8, 16, 32, 64),
    #     strides=(2, 2, 2, 2),
    # )

    # net = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    # )

    # net = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(32, 64, 128, 256, 512),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # )

    # # https://github.com/Project-MONAI/MONAI/discussions/3048
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    # net = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    # )

    # # https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_model.py
    # net = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(64, 128, 256, 512, 1024),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # )

    # # https://arxiv.org/pdf/1606.04797.pdf
    # net = monai.networks.nets.VNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     act = ("elu", {"inplace": True}),
    # )

    # net = monai.networks.nets.AttentionUnet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    # )

    # net = monai.networks.nets.UNETR(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     # feature_size=32, 
    #     img_size=(192, 192, 16),
    # )

    # net = monai.networks.nets.SwinUNETR(
    #     img_size=(128, 128, 32),
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    # )

    # net = monai.networks.nets.BasicUNetPlusPlus(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    # )

    # net = monai.networks.nets.SegResNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    # )

    # # https://github.com/naamiinepal/flare-2022/blob/36bbec2b767e2df0487427c20341d02d59e89697/configs/flare/c2f/version4.yaml#L44
    # net = monai.networks.nets.DynUNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     kernel_size=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    #     strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
    #     upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
    #     # act_name=relu,
    #     # norm_name=instance,
    # )
    # # https://github.com/UETAILab/echo-gan/blob/2774b91353a356d4bfb96568d6d23b5c52d09059/models/video_networks.py
    # # def Dynet():
    # #     sizes, spacings = [128, 128, 64], (1.5, 1.5, 1.5)
    # #     strides, kernels = [], []
    # #     while True:
    # #         spacing_ratio = [sp / min(spacings) for sp in spacings]
    # #         stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
    # #         kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
    # #         if all(s == 1 for s in stride):
    # #             break
    # #         sizes = [i / j for i, j in zip(sizes, stride)]
    # #         spacings = [i * j for i, j in zip(spacings, stride)]
    # #         kernels.append(kernel)
    # #         strides.append(stride)
    # #     strides.insert(0, len(spacings) * [1])
    # #     kernels.append(len(spacings) * [3])
    # net = monai.networks.nets.DynUNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    #     strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 1]],
    #     upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 1]],
    #     # act_name=relu,
    #     # norm_name=instance,
    # )

    # net = monai.networks.nets.FullyConnectedNet(
    #     in_channels=1,
    #     out_channels=num_classes,
    #     hidden_channels=[10, 20, 10],
    #     dropout=0.2
    # )

    # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth" 
    # to /home/liuhongzhi/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth
    # net = monai.networks.nets.FlexibleUNet(
    #         spatial_dims=3,
    #         in_channels = 1,
    #         out_channels = num_classes,
    #         backbone="efficientnet-b4",
    #         pretrained=True,
    # )

    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    # This is for SwinUNETR
    # patch_size = (128, 128, 32)   
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


def train(data_folder=".", model_folder="runs"):
    """run a training pipeline."""

    train_images = sorted(glob.glob(os.path.join(data_folder, "imagesTr", "*A1.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_folder, "labelsTr", "*A1.nii.gz")))
    test_images = sorted(glob.glob(os.path.join(data_folder, "imagesTs", "*B1.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(data_folder, "labelsTs", "*B1.nii.gz")))
    # train_images = sorted(glob.glob(os.path.join(data_folder, "imagesTr", "CuiYuYing_A1.nii.gz")))
    # train_labels = sorted(glob.glob(os.path.join(data_folder, "labelsTr", "CuiYuYing_A1.nii.gz")))
    # test_images = sorted(glob.glob(os.path.join(data_folder, "imagesTs", "cuishejun_A3.nii.gz")))
    # test_labels = sorted(glob.glob(os.path.join(data_folder, "labelsTs", "cuishejun_A3.nii.gz")))
    print("train_images: ", train_images)
    print("train_labels: ", train_labels)
    print("test_images: ", test_images)
    print("test_labels: ", test_labels)
    logging.info(f"train_images: {train_images}")
    logging.info(f"train_labels: {train_labels}")
    logging.info(f"test_images: {test_images}")
    logging.info(f"test_labels: {test_labels}")
    # logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")
    print(f"training: image/label ({len(train_images)}) testing: image/label ({len(test_images)}) folder: {data_folder}")
    logging.info(f"training: image/label ({len(train_images)}) testing: image/label ({len(test_images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    # train_frac, val_frac = 0.8, 0.2
    # n_train = int(train_frac * len(images)) + 1
    # n_val = min(len(images) - n_train, int(val_frac * len(images)))
    n_train = int(len(train_images)) + 1
    n_val = int(len(test_images)) + 1
    # logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    # train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(train_images[:n_train], train_labels[:n_train])]
    # val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(train_images, train_labels)]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(test_images, test_labels)]

    # create a training data loader
    batch_size = 2
    # batch_size = 1
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, 
                                       transform=train_transforms, 
                                       cache_rate=0.1) # , cache_rate=0.1)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, 
                                     transform=val_transforms, 
                                     cache_rate=0.1) #, cache_rate=0.1)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=8, # num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    nclasses = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net(num_classes=nclasses).to(device)
    max_epochs, lr, momentum = 500, 1e-4, 0.95 # 500, 1e-4, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # print("net: ", net)
    logging.info(f"model:\n{net}")

    # create evaluator (to be used to measure model quality during training  to_onehot=2 for 2 class
    # val_post_transform = monai.transforms.Compose(
    #     [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2)]   
    # )
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=nclasses)]
    )

    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        postprocessing=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.run()


def infer(data_folder=".", model_folder="runs", prediction_folder="output"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net(num_classes=4).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    # images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    images = sorted(glob.glob(os.path.join(image_folder, "imagesTs", "*.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    # saver = monai.transforms.SaveImage(output_dir=prediction_folder, mode="nearest", resample=True)
    saver = monai.transforms.SaveImage(
            output_dir=prediction_folder, 
            output_postfix='',
            separate_folder=False,
            mode="nearest", 
            resample=True
            )
    
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image'].meta['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            for p in preds:  # save each image+metadata in the batch respectively
                saver(p)

    # # copy the saved segmentations into the required folder structure for submission
    # submission_dir = os.path.join(prediction_folder, "to_submit")
    # if not os.path.exists(submission_dir):
    #     os.makedirs(submission_dir)
    # files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    # for f in files:
    #     new_name = os.path.basename(f)
    #     new_name = new_name[len("volume-covid19-A-0") :]
    #     new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
    #     to_name = os.path.join(submission_dir, new_name)
    #     shutil.copy(f, to_name)
    # logging.info(f"predictions copied to {submission_dir}.")


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
    model_folder = args.model_folder + "_" + curr_time
    os.makedirs(model_folder, exist_ok=True)

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
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
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("ankle_seg_data", "imagesTs")
        infer(data_folder=data_folder, model_folder=model_folder, prediction_folder=os.path.join(model_folder,"predict"))
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
