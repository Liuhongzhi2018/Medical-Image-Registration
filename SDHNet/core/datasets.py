import os
from os.path import join
import random
import numpy as np
import tifffile as tif

import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np


def read_datasets(path, datasets):
    files = []
    for d in datasets:
        files.extend([join(path, d, i) for i in os.listdir(join(path, d))])
    return files


def read_txt(path, file):
    with open(os.path.join(path, file), 'r') as f:
        content = f.readlines()
        # print(f"content: {content}")
    # filelist = [x.strip() for x in content if x.strip()]
    filelist = [x.strip() for x in content]
    # print(f"filelist: {filelist}")
    return filelist


#  filelist: ['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz 
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06_gt.nii.gz 
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz 
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01_gt.nii.gz',
# '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame09.nii.gz 
# /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame09_gt.nii.gz 
# /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame01.nii.gz 
# /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame01_gt.nii.gz', 


def generate_img_seg_pairs(filelist):
    # print(f"generate_imgpairs filelist: {filelist}")
    pairs = []
    for f in filelist:
        # print(f"f: {f}")
        pairs.append([f.split(' ')[0], f.split(' ')[1], f.split(' ')[2], f.split(' ')[3]])
    # print(f"pairs: {pairs}")
    return pairs

# f: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06_gt.nii.gz 
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz 
#  /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01_gt.nii.gz

# pairs: [['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz', 
# '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz'], 
# ['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame09.nii.gz', 
# '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame01.nii.gz'],

def generate_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
    return pairs


def generate_pairs_val(files):
    pairs = []
    labels = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
                labels.append([join(d1, 'segmentation.tif'), join(d2, 'segmentation.tif')])
    return pairs, labels


def generate_lspig_val(files):
    pairs = []
    labels = []
    files.sort()
    for i in range(0, len(files), 2):
        d1 = files[i]
        d2 = files[i + 1]
        pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
        labels.append([join(d1, 'segmentation.tif'), join(d2, 'segmentation.tif')])
        pairs.append([join(d2, 'volume.tif'), join(d1, 'volume.tif')])
        labels.append([join(d2, 'segmentation.tif'), join(d1, 'segmentation.tif')])

    return pairs, labels


def generate_atlas(atlas, files):
    pairs = []
    for d in files:
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])

    return pairs


def generate_atlas_val(atlas, files):
    pairs = []
    labels = []
    for d in files:
        if 'S01' in d:
            continue
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])
        labels.append([join(atlas, 'segmentation.tif'), join(d, 'segmentation.tif')])
    return pairs, labels


class ACDCTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        # self.size = [128, 128, 128]
        # self.datasets = ['abide', 'abidef', 'adhd', 'adni']
        # self.train_path = join(args.data_path, 'Train')
        self.train_path = join(args.data_path, args.dataset)
        # self.atlas = join(args.data_path, 'Test/lpba/S01')
        # self.files = read_datasets(self.train_path, self.datasets)
        self.files = read_txt(self.train_path, "train_img_seg_list.txt")
        # self.pairs = generate_atlas(self.atlas, self.files)
        self.pairs = generate_img_seg_pairs(self.files)
    
    def read_vol(self, filename):        
        img = nib.load(filename)  # shape: (Width-x, Height-y, Depth-z)
        vol = np.squeeze(img.dataobj)
        return vol

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        # index = index % len(self.pairs)
        index = index % len(self.files)
        # data1, data2 = self.pairs[index]
        mov, fixed = self.pairs[index][0], self.pairs[index][2]
        # print(f"data1: {data1} data2: {data2}")
        # data1: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame09.nii.gz 
        # data2: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient059/patient059_frame01.nii.gz

        # image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        # image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        image1 = torch.from_numpy(self.read_vol(mov)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(self.read_vol(fixed)[np.newaxis]).float() / 255.0
        # print(f"image1 shape: {image1.shape} image2 shape: {image2.shape}")
        # image1 shape: torch.Size([1, 256, 216, 9]) image2 shape: torch.Size([1, 256, 216, 9])

        return image1, image2

    def __len__(self):
        # return len(self.pairs)
        return len(self.files)


class ACDCTest(data.Dataset):
    def __init__(self, args):
        # self.size = [128, 128, 128]
        # self.datasets = [datas]
        self.test_path = join(args.data_path, args.dataset)
        # self.atlas = join(args.data_path, 'Test/lpba/S01')
        # self.files = read_datasets(self.test_path, self.datasets)
        self.files = read_txt(self.test_path, "test_img_seg_list.txt")
        # self.pairs, self.labels = generate_atlas_val(self.atlas, self.files)
        self.pairs = generate_img_seg_pairs(self.files)
        self.seg_values = np.unique(self.read_vol(self.pairs[0][3]))

    def read_vol(self, filename):        
        img = nib.load(filename)  # shape: (Width-x, Height-y, Depth-z)
        vol = np.squeeze(img.dataobj)
        return vol

    def __getitem__(self, index):
        mov, fixed = self.pairs[index][0], self.pairs[index][2]
        mov_seg, fixed_seg = self.pairs[index][1], self.pairs[index][3]
        print(f"data1: {mov} data2: {fixed} seg1: {mov_seg} seg2: {fixed_seg}")

        # image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        # image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0
        image1 = torch.from_numpy(self.read_vol(mov)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(self.read_vol(fixed)[np.newaxis]).float() / 255.0

        # label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        # label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()
        label1 = torch.from_numpy(self.read_vol(mov_seg)[np.newaxis]).float()
        label2 = torch.from_numpy(self.read_vol(fixed_seg)[np.newaxis]).float()

        return mov, image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class LiverTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [128, 128, 128]
        self.datasets = ['msd_hep', 'msd_pan', 'msd_liver', 'youyi_liver']
        self.train_path = join(args.data_path, 'Train')
        self.files = read_datasets(self.train_path, self.datasets)
        self.pairs = generate_pairs(self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class LiverTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_pairs_val(self.files)

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class LspigTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_lspig_val(self.files)

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


class BrainTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [128, 128, 128]
        self.datasets = ['abide', 'abidef', 'adhd', 'adni']
        self.train_path = join(args.data_path, 'Train')
        self.atlas = join(args.data_path, 'Test/lpba/S01')
        self.files = read_datasets(self.train_path, self.datasets)
        self.pairs = generate_atlas(self.atlas, self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        index = index % len(self.pairs)
        data1, data2 = self.pairs[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        return image1, image2

    def __len__(self):
        return len(self.pairs)


class BrainTest(data.Dataset):
    def __init__(self, args, datas):
        self.size = [128, 128, 128]
        self.datasets = [datas]
        self.test_path = join(args.data_path, 'Test')
        self.atlas = join(args.data_path, 'Test/lpba/S01')
        self.files = read_datasets(self.test_path, self.datasets)
        self.pairs, self.labels = generate_atlas_val(self.atlas, self.files)
        self.seg_values = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                           50, 61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102,
                           121, 122, 161, 162, 163, 164, 165, 166, 181, 182]

    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]

        image1 = torch.from_numpy(tif.imread(data1)[np.newaxis]).float() / 255.0
        image2 = torch.from_numpy(tif.imread(data2)[np.newaxis]).float() / 255.0

        label1 = torch.from_numpy(tif.imread(seg1)[np.newaxis]).float()
        label2 = torch.from_numpy(tif.imread(seg2)[np.newaxis]).float()

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)


