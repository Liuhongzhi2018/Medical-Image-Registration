import os
from os.path import join
import random
import numpy as np
import tifffile as tif
import SimpleITK as sitk
import torch
import torch.utils.data as data
import torch.nn.functional as F

def load_nii(path):
    nii = sitk.ReadImage(path)
    nii_array = sitk.GetArrayFromImage(nii)
    # print(f"load_nii GetSize: {nii.GetSize()} shape: {nii_array.shape}")
    # load_nii GetSize: (160, 224, 192) WHD shape: (192, 224, 160) Depth, Height, Width
    return nii_array

def load_nii_GetSpacing(path):
    nii = sitk.ReadImage(path)
    nii_array = sitk.GetArrayFromImage(nii)
    # print(f"load_nii GetSize: {nii.GetSize()} shape: {nii_array.shape}")
    # load_nii GetSize: (160, 224, 192) WHD shape: (192, 224, 160) Depth, Height, Width
    nii_spacing = nii.GetSpacing()
    return nii_array, nii_spacing

def read_datasets(path, datasets):
    files = []
    for d in datasets:
        files.extend([join(path, d, i) for i in os.listdir(join(path, d))])
    return files


def generate_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
    return pairs[:270]


def generate_pairs_val(files):
    pairs = []
    labels = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'volume.tif'), join(d2, 'volume.tif')])
                labels.append([join(d1, 'segmentation.tif'), join(d2, 'segmentation.tif')])
    return pairs[270:], labels[270:]


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
        if 'S01' in d:
            continue
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])

    return pairs[:27]


def generate_atlas_val(atlas, files):
    pairs = []
    labels = []
    for d in files:
        if 'S01' in d:
            continue
        pairs.append([join(atlas, 'volume.tif'), join(d, 'volume.tif')])
        labels.append([join(atlas, 'segmentation.tif'), join(d, 'segmentation.tif')])
    return pairs[27:], labels[27:]

def generate_oasis_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([join(d1, 'aligned_norm.nii.gz'), join(d2, 'aligned_norm.nii.gz')])
    return pairs


def generate_oasis_pairs_val(files):
    pairs = []
    labels = []
    for i in range(len(files)-1):
        pairs.append([join(files[i], 'aligned_norm.nii.gz'), join(files[i+1], 'aligned_norm.nii.gz')])
        labels.append([join(files[i], 'aligned_seg35.nii.gz'), join(files[i+1], 'aligned_seg35.nii.gz')])
    return pairs, labels


def read_txt(path, file):
    with open(os.path.join(path, file), 'r') as f:
        content = f.readlines()
        # print(f"content: {content}")
    # filelist = [x.strip() for x in content if x.strip()]
    filelist = [x.strip() for x in content]
    # print(f"filelist: {filelist}")
    return filelist


def generate_img_seg_pairs(filelist):
    # print(f"generate_imgpairs filelist: {filelist}")
    pairs = []
    for f in filelist:
        # print(f"f: {f}")
        fline = f.split(' ')
        # print(f"generate_img_seg_pairs fline {fline}")
        # generate_img_seg_pairs fline 
        # ['/mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0002_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0002_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0001_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0001_0000.nii.gz']
        pairs.append([fline[0], fline[1], fline[2], fline[3]])
    # print(f"pairs: {pairs}")
    return pairs


class ACDCTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_train']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs = generate_oasis_pairs(self.files)  # 270 (first 270 of 342 pairs)
        self.train_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.train_path, "train_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.ymax, self.ymin = 255, 0

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        # print(f"ACDCTrain pairs {self.pairs}")
        # ACDCTrain pairs [['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06_gt.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01_gt.nii.gz']]
        index = index % len(self.pairs)
        # data1, data2 = self.pairs[index]
        moving, fixed = self.pairs[index][0], self.pairs[index][2]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving)[np.newaxis]).float()
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTrain fixed image1: torch.Size([1, 17, 224, 154]) moving image2: torch.Size([1, 17, 224, 154])
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTrain fixed max: 255.0 min: 3.0 moving max: 255.0 min: 3.0
        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        return image1, image2

    def __len__(self):
        return len(self.pairs)


class ACDCTest(data.Dataset):
    def __init__(self, args):
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_val']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs, self.labels = generate_oasis_pairs_val(self.files)  # 72 (last 72 of 342 pairs)
        self.test_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.test_path, "test_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.seg_values = [i for i in range(1, 4)]
        self.ymax, self.ymin = 255, 0

    def __getitem__(self, index):
        moving_img, fixed_img = self.pairs[index][0], self.pairs[index][2]
        moving_seg, fixed_seg = self.pairs[index][1], self.pairs[index][3]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed_img)[np.newaxis]).float()
        # image1 = torch.from_numpy(load_nii_GetSpacing(fixed_img)[0][np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving_img)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii_GetSpacing(moving_img)[0][np.newaxis]).float()

        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTest fixed image1: torch.Size([1, 16, 174, 208]) moving image2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTest fixed max: 255.0 min: 0.0 moving max: 255.0 min: 0.0

        # label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label1 = torch.from_numpy(load_nii(fixed_seg)[np.newaxis]).float()
        # label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(moving_seg)[np.newaxis]).float()
        # label1 = F.interpolate(label1.float(), size=self.size, mode='trilinear')
        # label2 = F.interpolate(label2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed label1: {label1.shape} moving label2: {label2.shape}")
        # ACDCTest fixed label1: torch.Size([1, 16, 174, 208]) moving label2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {label1.max()} min: {label1.min()} moving max: {label2.max()} min: {label2.min()}")
        # ACDCTest fixed max: 3.0 min: 0.0 moving max: 3.0 min: 0.0

        return image1, image2, label1, label2, moving_img

    def __len__(self):
        return len(self.pairs)


class OAIZIBTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_train']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs = generate_oasis_pairs(self.files)  # 270 (first 270 of 342 pairs)
        self.train_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.train_path, "train_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.ymax, self.ymin = 255, 0

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        # print(f"ACDCTrain pairs {self.pairs}")
        # ACDCTrain pairs [['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06_gt.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01_gt.nii.gz']]
        index = index % len(self.pairs)
        # data1, data2 = self.pairs[index]
        moving, fixed = self.pairs[index][0], self.pairs[index][2]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving)[np.newaxis]).float()
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTrain fixed image1: torch.Size([1, 17, 224, 154]) moving image2: torch.Size([1, 17, 224, 154])
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTrain fixed max: 255.0 min: 3.0 moving max: 255.0 min: 3.0
        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        return image1, image2

    def __len__(self):
        return len(self.pairs)


class OAIZIBTest(data.Dataset):
    def __init__(self, args):
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_val']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs, self.labels = generate_oasis_pairs_val(self.files)  # 72 (last 72 of 342 pairs)
        self.test_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.test_path, "test_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.seg_values = [i for i in range(1, 6)]
        self.ymax, self.ymin = 255, 0

    def __getitem__(self, index):
        moving_img, fixed_img = self.pairs[index][0], self.pairs[index][2]
        moving_seg, fixed_seg = self.pairs[index][1], self.pairs[index][3]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed_img)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving_img)[np.newaxis]).float()
        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTest fixed image1: torch.Size([1, 16, 174, 208]) moving image2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTest fixed max: 255.0 min: 0.0 moving max: 255.0 min: 0.0

        # label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label1 = torch.from_numpy(load_nii(fixed_seg)[np.newaxis]).float()
        # label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(moving_seg)[np.newaxis]).float()
        # label1 = F.interpolate(label1.float(), size=self.size, mode='trilinear')
        # label2 = F.interpolate(label2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed label1: {label1.shape} moving label2: {label2.shape}")
        # ACDCTest fixed label1: torch.Size([1, 16, 174, 208]) moving label2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {label1.max()} min: {label1.min()} moving max: {label2.max()} min: {label2.min()}")
        # ACDCTest fixed max: 3.0 min: 0.0 moving max: 3.0 min: 0.0
        # print(f"ACDCTest fixed: {torch.unique(label1)} moving: {torch.unique(label2)}")
        # ACDCTest fixed: tensor([0, 1, 2, 3, 4, 5.]) moving: tensor([0, 1, 2, 3, 4, 5.])
        return image1, image2, label1, label2, moving_img

    def __len__(self):
        return len(self.pairs)


class LPBATrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_train']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs = generate_oasis_pairs(self.files)  # 270 (first 270 of 342 pairs)
        self.train_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.train_path, "train_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.ymax, self.ymin = 255, 0

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        # print(f"ACDCTrain pairs {self.pairs}")
        # ACDCTrain pairs [['/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame06_gt.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient075/patient075_frame01_gt.nii.gz']]
        index = index % len(self.pairs)
        # data1, data2 = self.pairs[index]
        moving, fixed = self.pairs[index][0], self.pairs[index][2]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving)[np.newaxis]).float()
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTrain fixed image1: torch.Size([1, 17, 224, 154]) moving image2: torch.Size([1, 17, 224, 154])
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTrain fixed max: 255.0 min: 3.0 moving max: 255.0 min: 3.0
        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # print(f"ACDCTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        return image1, image2

    def __len__(self):
        return len(self.pairs)


class LPBATest(data.Dataset):
    def __init__(self, args):
        self.size = [192, 224, 160]
        # self.datasets = ['oasis_val']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs, self.labels = generate_oasis_pairs_val(self.files)  # 72 (last 72 of 342 pairs)
        self.test_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.test_path, "test_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.ymax, self.ymin = 255, 0
        self.seg_values = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49,
         50, 61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83,
         84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121,
        122, 161, 162, 163, 164, 165, 166, 181, 182]

    def __getitem__(self, index):
        moving_img, fixed_img = self.pairs[index][0], self.pairs[index][2]
        moving_seg, fixed_seg = self.pairs[index][1], self.pairs[index][3]

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed_img)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving_img)[np.newaxis]).float()
        image1 = (image1 - image1.min()) * (self.ymax - self.ymin)  / (image1.max() - image1.min()) + self.ymin
        image2 = (image2 - image2.min()) * (self.ymax - self.ymin)  / (image2.max() - image2.min()) + self.ymin
        # image1 = F.interpolate(image1.float(), size=self.size, mode='trilinear')
        # image2 = F.interpolate(image2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed image1: {image1.shape} moving image2: {image2.shape}")
        # ACDCTest fixed image1: torch.Size([1, 16, 174, 208]) moving image2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # ACDCTest fixed max: 255.0 min: 0.0 moving max: 255.0 min: 0.0

        # label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label1 = torch.from_numpy(load_nii(fixed_seg)[np.newaxis]).float()
        # label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(moving_seg)[np.newaxis]).float()
        # label1 = F.interpolate(label1.float(), size=self.size, mode='trilinear')
        # label2 = F.interpolate(label2.float(), size=self.size, mode='trilinear')
        # print(f"ACDCTest fixed label1: {label1.shape} moving label2: {label2.shape}")
        # ACDCTest fixed label1: torch.Size([1, 16, 174, 208]) moving label2: torch.Size([1, 16, 174, 208])
        # print(f"ACDCTest fixed max: {label1.max()} min: {label1.min()} moving max: {label2.max()} min: {label2.min()}")
        # ACDCTest fixed max: 3.0 min: 0.0 moving max: 3.0 min: 0.0
        # print(f"ACDCTest fixed: {torch.unique(label1)} moving: {torch.unique(label2)}")
        # ACDCTest fixed: tensor([0, 1, 2, 3, 4, 5.]) moving: tensor([0, 1, 2, 3, 4, 5.])
        # print(f"LPBATest fixed: {torch.unique(label1)} moving: {torch.unique(label2)}")
        return image1, image2, label1, label2, moving_img

    def __len__(self):
        return len(self.pairs)


class OASISTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [160, 192, 224]
        # self.datasets = ['oasis_train']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs = generate_oasis_pairs(self.files)  # 270 (first 270 of 342 pairs)
        self.train_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.train_path, "train_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)

    def __getitem__(self, index):
        if not self.seed:
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            self.seed = True

        # print(f"OASISTrain pairs {self.pairs}")
        # [['/mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0395_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0395_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0394_0000.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0394_0000.nii.gz']]

        index = index % len(self.pairs)
        # data1, data2 = self.pairs[index]
        moving, fixed = self.pairs[index][0], self.pairs[index][2]
        # print(f"fixed: {fixed} moving: {moving}")
        # fixed: /mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0007_0000.nii.gz 
        # moving: /mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0008_0000.nii.gz

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving)[np.newaxis]).float()
        # print(f"OASISTrain fixed image1: {image1.shape} moving image2: {image2.shape}")
        # OASISTrain fixed image1: torch.Size([1, 192, 224, 160]) moving image2: torch.Size([1, 192, 224, 160])
        # print(f"OASISTrain fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # OASISTrain fixed max: 0.8352941274642944 min: 0.0 moving max: 0.9058823585510254 min: 0.0
        return image1, image2

    def __len__(self):
        return len(self.pairs)


class OASISTest(data.Dataset):
    def __init__(self, args):
        self.size = [160, 192, 224]
        # self.datasets = ['oasis_val']
        # self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        # self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
        # self.pairs, self.labels = generate_oasis_pairs_val(self.files)  # 72 (last 72 of 342 pairs)
        self.test_path = join(args.data_path, args.dataset)
        self.files = read_txt(self.test_path, "test_img_seg_list.txt")
        self.pairs = generate_img_seg_pairs(self.files)
        self.seg_values = [i for i in range(1, 36)]

    def __getitem__(self, index):
        moving_img, fixed_img = self.pairs[index][0], self.pairs[index][2]
        moving_seg, fixed_seg = self.pairs[index][1], self.pairs[index][3]
        # print(f"moving_img: {moving_img} fixed_img: {fixed_img}")
        # moving_img: /mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0396_0000.nii.gz 
        # fixed_img: /mnt/lhz/Datasets/Learn2reg/OASIS/imagesTr/OASIS_0395_0000.nii.gz
        # print(f"moving_seg: {moving_seg} fixed_seg: {fixed_seg}")
        # moving_seg: /mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0396_0000.nii.gz 
        # fixed_seg: /mnt/lhz/Datasets/Learn2reg/OASIS/labelsTr/OASIS_0395_0000.nii.gz

        # image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
        image1 = torch.from_numpy(load_nii(fixed_img)[np.newaxis]).float()
        # image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()
        image2 = torch.from_numpy(load_nii(moving_img)[np.newaxis]).float()
        # print(f"OASISTest fixed image1: {image1.shape} moving image2: {image2.shape}")
        # OASISTest fixed image1: torch.Size([1, 192, 224, 160]) moving image2: torch.Size([1, 192, 224, 160])
        # print(f"OASISTest fixed max: {image1.max()} min: {image1.min()} moving max: {image2.max()} min: {image2.min()}")
        # OASISTest fixed max: 0.7882353067398071 min: 0.0 moving max: 0.8980392217636108 min: 0.0

        # label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
        label1 = torch.from_numpy(load_nii(fixed_seg)[np.newaxis]).float()
        # label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()
        label2 = torch.from_numpy(load_nii(moving_seg)[np.newaxis]).float()
        # print(f"OASISTest fixed label1: {label1.shape} moving label2: {label2.shape}")
        # OASISTest fixed label1: torch.Size([1, 192, 224, 160]) moving label2: torch.Size([1, 192, 224, 160])
        # print(f"OASISTest fixed max: {label1.max()} min: {label1.min()} moving max: {label2.max()} min: {label2.min()}")
        # OASISTest fixed max: 35.0 min: 0.0 moving max: 35.0 min: 0.0
        return image1, image2, label1, label2, moving_img

    def __len__(self):
        return len(self.pairs)


# class OasisTrain(data.Dataset):
#     def __init__(self, args):
#         self.seed = False
#         self.size = [160, 192, 224]
#         self.datasets = ['oasis_train']
#         self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
#         self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
#         self.pairs = generate_oasis_pairs(self.files)  # 270 (first 270 of 342 pairs)

#     def __getitem__(self, index):
#         if not self.seed:
#             random.seed(123)
#             np.random.seed(123)
#             torch.manual_seed(123)
#             torch.cuda.manual_seed_all(123)
#             self.seed = True

#         index = index % len(self.pairs)
#         data1, data2 = self.pairs[index]

#         image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
#         image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

#         return image1, image2

#     def __len__(self):
#         return len(self.pairs)


# class OasisTest(data.Dataset):
#     def __init__(self, args):
#         self.size = [160, 192, 224]
#         self.datasets = ['oasis_val']
#         self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
#         self.files = sorted(self.files, key=lambda x: int(x[-8:-4]))
#         self.pairs, self.labels = generate_oasis_pairs_val(self.files)  # 72 (last 72 of 342 pairs)
#         self.seg_values = [i for i in range(1, 36)]

#     def __getitem__(self, index):
#         data1, data2 = self.pairs[index]
#         seg1, seg2 = self.labels[index]

#         image1 = torch.from_numpy(load_nii(data1)[np.newaxis]).float()
#         image2 = torch.from_numpy(load_nii(data2)[np.newaxis]).float()

#         label1 = torch.from_numpy(load_nii(seg1)[np.newaxis]).float()
#         label2 = torch.from_numpy(load_nii(seg2)[np.newaxis]).float()

#         return image1, image2, label1, label2

#     def __len__(self):
#         return len(self.pairs)

class LiverTrain(data.Dataset):
    def __init__(self, args):
        self.seed = False
        self.size = [128, 128, 128]
        self.datasets = ['sliver_val']
        self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        self.pairs = generate_pairs(self.files)  # 270 (first 270 of 342 pairs)

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
        self.files = read_datasets(args.data_path, self.datasets)  # sliver 342 = 19 * 18
        self.pairs, self.labels = generate_pairs_val(self.files)  # 72 (last 72 of 342 pairs)

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
        self.datasets = [datas]  # lspig 17
        self.files = read_datasets(args.data_path, self.datasets)
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
        self.datasets = ['lpba_val']
        self.atlas = args.data_path + 'lpba_val/S01'
        self.files = read_datasets(args.data_path, self.datasets)  # lpba 39 = 1 * 39
        self.pairs = generate_atlas(self.atlas, self.files)  # 27 (first 27 of 39 pairs)

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
        self.atlas = args.data_path + 'lpba_val/S01'
        self.files = read_datasets(args.data_path, self.datasets)  # # lpba 39 = 1 * 39
        self.pairs, self.labels = generate_atlas_val(self.atlas, self.files)  # 12 (last 12 of 39 pairs)
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
