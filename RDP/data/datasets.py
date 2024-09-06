import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch.nn.functional as F
import nibabel as nib

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class OAIZIBDataset(Dataset):
    def __init__(self, fn, transforms):
        self.train_file = fn
        self.transforms = transforms
        self.train_file_list = self.read_file_tolist(self.train_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):
        # print(f"LPBADataset read_vol: {filename}")
        # OAIZIBDataset read_vol: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
        
    def __getitem__(self, index):
        f_index = index % (len(self.train_file_list))
        f_paths = self.train_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]

        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)  
        # print(f"OAIZIBDataset x shape: {x.shape} y shape: {y.shape}")
        # OAIZIBDataset x shape: (160, 384, 384) y shape: (160, 384, 384)
        # print(f"OAIZIBDataset x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # OAIZIBDataset x seg shape: [0 1 2 3 4 5] y seg shape: [0 1 2 3 4 5]

        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        
        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"OAIZIBDataset out x shape: {x.shape} y shape: {y.shape}")
        # OAIZIBDataset out x shape: torch.Size([1, 160, 384, 384]) y shape: torch.Size([1, 160, 384, 384])
        return f_name, x, y

    def __len__(self):
        return len(self.train_file_list)


class OAIZIBDatasetVal(Dataset):
    def __init__(self, fn, transforms):
        self.val_file = fn
        self.transforms = transforms
        self.val_file_list = self.read_file_tolist(self.val_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):        
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
    
    def __getitem__(self, index):
        f_index = index % (len(self.val_file_list))
        f_paths = self.val_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]
        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)
        # print(f"OAIZIBDatasetVal x shape: {x.shape} y shape: {y.shape}")
        # OAIZIBDatasetVal x shape: (160, 384, 384) y shape: (160, 384, 384)
        # print(f"OAIZIBDatasetVal x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # OAIZIBDatasetVal x seg shape: [0 1 2 3 4 5] y seg shape: [0 1 2 3 4 5]

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        # return f_name, x, y, x_seg, y_seg
        return f_paths[0], x, y, x_seg, y_seg

    def __len__(self):
        return len(self.val_file_list)
    

class OASISDataset(Dataset):
    def __init__(self, fn, transforms):
        self.train_file = fn
        self.transforms = transforms
        self.train_file_list = self.read_file_tolist(self.train_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):
        # print(f"LPBADataset read_vol: {filename}")
        # LPBADataset read_vol: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
        
    def __getitem__(self, index):
        f_index = index % (len(self.train_file_list))
        f_paths = self.train_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]

        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)  
        # print(f"OASISDataset x shape: {x.shape} y shape: {y.shape}")
        # OASISDataset x shape: (160, 224, 192) y shape: (160, 224, 192)
        # print(f"OASISDataset x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # OASISDataset x seg shape: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
        #                            21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
        
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        
        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"OASISDataset out x shape: {x.shape} y shape: {y.shape}")
        # OASISDataset out x shape: torch.Size([1, 160, 224, 192]) y shape: torch.Size([1, 160, 224, 192])
        return f_name, x, y

    def __len__(self):
        return len(self.train_file_list)
    
class OASISDatasetVal(Dataset):
    def __init__(self, fn, transforms):
        self.val_file = fn
        self.transforms = transforms
        self.val_file_list = self.read_file_tolist(self.val_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):        
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
    
    def __getitem__(self, index):
        f_index = index % (len(self.val_file_list))
        f_paths = self.val_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]
        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)
        # print(f"OASISDatasetVal x shape: {x.shape} y shape: {y.shape}")
        # OASISDatasetVal x shape: (160, 224, 192) y shape: (160, 224, 192)
        # print(f"OASISDatasetVal x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # OASISDatasetVal x seg shape: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #                               24 25 26 27 28 29 30 31 32 33 34 35]
        # y seg shape: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #               24 25 26 27 28 29 30 31 32 33 34 35]
        
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        # return f_name, x, y, x_seg, y_seg
        return f_paths[0], x, y, x_seg, y_seg

    def __len__(self):
        return len(self.val_file_list)

class ACDCDataset(Dataset):
    def __init__(self, fn, transforms):
        self.train_file = fn
        self.transforms = transforms
        self.train_file_list = self.read_file_tolist(self.train_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):
        # print(f"LPBADataset read_vol: {filename}")
        # LPBADataset read_vol: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
        
    def __getitem__(self, index):
        f_index = index % (len(self.train_file_list))
        f_paths = self.train_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]

        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)  
        # print(f"ACDCDataset x shape: {x.shape} y shape: {y.shape}")
        # ACDCDataset x shape: (216, 256, 10) y shape: (216, 256, 10)
        # print(f"ACDCDataset x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # ACDCDataset x seg shape: [0 1 2 3] y seg shape: [0 1 2 3]
        
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        
        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"ACDCDataset out x shape: {x.shape} y shape: {y.shape}")
        # ACDCDataset out x shape: torch.Size([1, 216, 256, 10]) y shape: torch.Size([1, 216, 256, 10])
        return f_name, x, y

    def __len__(self):
        return len(self.train_file_list)
    
class ACDCDatasetVal(Dataset):
    def __init__(self, fn, transforms):
        self.val_file = fn
        self.transforms = transforms
        self.val_file_list = self.read_file_tolist(self.val_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):        
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
    
    def __getitem__(self, index):
        f_index = index % (len(self.val_file_list))
        f_paths = self.val_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]
        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        # return f_name, x, y, x_seg, y_seg
        return f_paths[0], x, y, x_seg, y_seg

    def __len__(self):
        return len(self.val_file_list)

    
class LPBADataset(Dataset):
    def __init__(self, fn, transforms):
        self.train_file = fn
        self.transforms = transforms
        self.train_file_list = self.read_file_tolist(self.train_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):
        # print(f"LPBADataset read_vol: {filename}")
        # LPBADataset read_vol: /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
        
    def __getitem__(self, index):
        f_index = index % (len(self.train_file_list))
        f_paths = self.train_file_list[f_index].split(" ")
        # print(f"LPBADataset __getitem__: {f_paths}")
        # LPBADataset __getitem__: [
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10_gt.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame01.nii.gz', 
        # '/mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame01_gt.nii.gz']

        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]
        # print(f"LPBADataset f_path: {x} {x_seg} {y} {y_seg}")
        # LPBADataset f_path: 
        # /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10.nii.gz 
        # /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame10_gt.nii.gz 
        # /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame01.nii.gz 
        # /mnt/lhz/Datasets/Learn2reg/ACDC/training/patient087/patient087_frame01_gt.nii.gz

        x = self.read_vol(x)
        # x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        # y_seg = self.read_vol(y_seg)  
        # print(f"LPBADataset x shape: {x.shape} y shape: {y.shape}")
        # LPBADataset x shape: (256, 184, 8) y shape: (256, 184, 8)
        
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(f"LPBADataset x None shape: {x.shape} y shape: {y.shape}")
        # LPBADataset x None shape: (1, 256, 184, 8) y shape: (1, 256, 184, 8)
        x, y = self.transforms([x, y])
        # print(f"LPBADataset transforms x shape: {x.shape} y shape: {y.shape}")
        # LPBADataset transforms x shape: (1, 256, 184, 8) y shape: (1, 256, 184, 8)
        
        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        # print(f"LPBADataset x np shape: {x.shape} y shape: {y.shape}")
        # LPBADataset x np shape: (1, 256, 184, 8) y shape: (1, 256, 184, 8)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"LPBADataset out x shape: {x.shape} y shape: {y.shape}")
        # LPBADataset out x shape: torch.Size([1, 256, 184, 8]) y shape: torch.Size([1, 256, 184, 8])
        return x, y

    def __len__(self):
        # return len(self.train_file_list)*(len(self.train_file_list)-1)
        return len(self.train_file_list)
    
    
class LPBADatasetVal(Dataset):
    def __init__(self, fn, transforms):
        self.val_file = fn
        self.transforms = transforms
        self.val_file_list = self.read_file_tolist(self.val_file)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def read_file_tolist(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
        filelist = [x.strip() for x in content if x.strip()]
        return filelist
    
    def read_vol(self, filename):        
        img = nib.load(filename)
        vol = np.squeeze(img.dataobj)
        return vol
    
    def translabel(self, img):
        seg_table = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 
                     66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
                     101, 102, 121, 122, 161, 162, 163, 164, 165, 166, 181, 182]
        img_out = np.zeros_like(img)
        for i in range(len(seg_table)):
            img_out[img == seg_table[i]] = i
        return img_out

    def __getitem__(self, index):
        f_index = index % (len(self.val_file_list))
        f_paths = self.val_file_list[f_index].split(" ")
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]
        x = self.read_vol(x)
        x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        y_seg = self.read_vol(y_seg)
        # print(f"LPBADatasetVal {f_name} x shape: {x.shape} x_seg : {x_seg.shape} y shape: {y.shape} y_seg: {y_seg.shape}")
        # LPBADatasetVal S07.delineation.skullstripped.nii.gz x shape: (160, 192, 160) x_seg : (160, 192, 160) y shape: (160, 192, 160) y_seg: (160, 192, 160)
        # print(f"LPBADatasetVal {f_name} x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        x_seg = self.translabel(x_seg)
        y_seg = self.translabel(y_seg)
        # print(f"LPBADatasetVal translabel {f_name} x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        # LPBADatasetVal translabel S06.delineation.skullstripped.nii.gz 
        # x_seg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
        # 48 49 50 51 52 53 54 55 56] 
        # y_seg : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
        # 48 49 50 51 52 53 54 55 56]

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(f"LPBADatasetVal {f_name} x None shape: {x.shape} y shape: {y.shape}")
        # LPBADatasetVal S07.delineation.skullstripped.nii.gz x None shape: (1, 160, 192, 160) y shape: (1, 160, 192, 160)
        # print(f"LPBADatasetVal {f_name} None x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        # print(f"LPBADatasetVal {f_name} transforms x shape: {x.shape} x_seg shape: {x_seg.shape} y shape: {y.shape} y_seg: {y_seg.shape}")
        # LPBADatasetVal S08.delineation.skullstripped.nii.gz transforms x shape: (1, 160, 192, 160) x_seg shape: (1, 160, 192, 160) y shape: (1, 160, 192, 160) y_seg: (1, 160, 192, 160)
        # print(f"LPBADatasetVal {f_name} transforms x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        
        # [Bsize,channelsHeight,,Width,Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        # print(f"LPBADatasetVal {f_name} x np shape: {x.shape} x_seg shape: {x_seg.shape} y shape: {y.shape} y_seg shape: {y_seg.shape}")
        # LPBADatasetVal S08.delineation.skullstripped.nii.gz x np shape: (1, 160, 192, 160) x_seg shape: (1, 160, 192, 160) y shape: (1, 160, 192, 160) y_seg shape: (1, 160, 192, 160)
        
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        # print(f"LPBADatasetVal {f_name} out x shape: {x.shape} x_seg shape: {x_seg.shape} y shape: {y.shape} y_seg shape: {y_seg.shape}")
        # LPBADatasetVal S08.delineation.skullstripped.nii.gz out x shape: torch.Size([1, 160, 192, 160]) x_seg shape: torch.Size([1, 160, 192, 160]) y shape: torch.Size([1, 160, 192, 160]) y_seg shape: torch.Size([1, 160, 192, 160])
        # print(f"LPBADatasetVal {f_name} unique x_seg: {torch.unique(x_seg)} y_seg : {torch.unique(y_seg)}")
        # LPBADatasetVal S09.delineation.skullstripped.nii.gz unique 
        # x_seg: tensor([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
        #  34,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  61,  62,  63,
        #  64,  65,  66,  67,  68,  81,  82,  83,  84,  85,  86,  87,  88,  89,
        #  90,  91,  92, 101, 102, 121, 122, 161, 162, 163, 164, 165, 166, 181,
        # 182], dtype=torch.int16) 
        # y_seg : tensor([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
        #  34,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  61,  62,  63,
        #  64,  65,  66,  67,  68,  81,  82,  83,  84,  85,  86,  87,  88,  89,
        #  90,  91,  92, 101, 102, 121, 122, 161, 162, 163, 164, 165, 166, 181,
        # 182], dtype=torch.int16)

        return f_name, x, y, x_seg, y_seg

    def __len__(self):
        # return len(self.train_file_list)*(len(self.train_file_list)-1)
        return len(self.val_file_list)

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        # print(f"LPBADataset x shape: {x.shape} y: {y.shape}")
        # LPBADataset x shape: (160, 192, 160) y: (160, 192, 160)
        # print(f"LPBABrainDatasetS2S x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        # LPBABrainDatasetS2S x_seg: [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43
        #   44  45  46  47  48  49  50  61  62  63  64  65  66  67  68  81  82  83
        #   84  85  86  87  88  89  90  91  92 101 102 121 122 161 162 163 164 165
        #  166 181 182] 
        # y_seg : [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43
        #   44  45  46  47  48  49  50  61  62  63  64  65  66  67  68  81  82  83
        #   84  85  86  87  88  89  90  91  92 101 102 121 122 161 162 163 164 165
        #  166 181 182]
        
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(f"LPBADataset x None shape: {x.shape} y: {y.shape}")
        # LPBADataset x None shape: (1, 160, 192, 160) y: (1, 160, 192, 160)
        
        x, y = self.transforms([x, y])
        # print(f"LPBADataset transforms x shape: {x.shape} y: {y.shape}")
        # LPBADataset transforms x shape: (1, 160, 192, 160) y: (1, 160, 192, 160)
        
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # print(f"LPBADataset x np shape: {x.shape} y: {y.shape}")
        # LPBADataset x np shape: (1, 160, 192, 160) y: (1, 160, 192, 160)
        
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"LPBADataset out x shape: {x.shape} y: {y.shape}")
        # LPBADataset out x shape: torch.Size([1, 160, 192, 160]) y: torch.Size([1, 160, 192, 160])
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # print(os.path.basename(path_x), os.path.basename(path_y))
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


# class LPBABrainHalfDatasetS2S(Dataset):
#     def __init__(self, data_path, transforms):
#         self.paths = data_path
#         self.transforms = transforms

#     def one_hot(self, img, C):
#         out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
#         for i in range(C):
#             out[i,...] = img == i
#         return out
#     def half_pair(self,pair):
#         return pair[0][::2,::2,::2], pair[1][::2,::2,::2]

#     def __getitem__(self, index):
#         x_index = index // (len(self.paths) - 1)
#         s = index % (len(self.paths) - 1)
#         y_index = s + 1 if s >= x_index else s
#         path_x = self.paths[x_index]
#         path_y = self.paths[y_index]
#         x, x_seg = self.half_pair(pkload(path_x))
#         y, y_seg = self.half_pair(pkload(path_y))

#         #print(x.shape)
#         #print(x.shape)
#         #print(np.unique(y))
#         # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
#         # transforms work with nhwtc
#         x, y = x[None, ...], y[None, ...]
#         # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
#         x,y = self.transforms([x, y])
#         #y = self.one_hot(y, 2)
#         #print(y.shape)
#         #sys.exit(0)
#         x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
#         y = np.ascontiguousarray(y)
#         #plt.figure()
#         #plt.subplot(1, 2, 1)
#         #plt.imshow(x[0, :, :, 8], cmap='gray')
#         #plt.subplot(1, 2, 2)
#         #plt.imshow(y[0, :, :, 8], cmap='gray')
#         #plt.show()
#         #sys.exit(0)
#         #y = np.squeeze(y, axis=0)
#         x, y = torch.from_numpy(x), torch.from_numpy(y)
#         return x, y

#     def __len__(self):
#         return len(self.paths)*(len(self.paths)-1)


# class LPBABrainHalfInferDatasetS2S(Dataset):
#     def __init__(self, data_path, transforms):
#         self.paths = data_path
#         self.transforms = transforms

#     def one_hot(self, img, C):
#         out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
#         for i in range(C):
#             out[i,...] = img == i
#         return out
#     def half_pair(self,pair):
#         return pair[0][::2,::2,::2], pair[1][::2,::2,::2]
#     def __getitem__(self, index):
#         x_index = index//(len(self.paths)-1)
#         s = index%(len(self.paths)-1)
#         y_index = s+1 if s >= x_index else s
#         path_x = self.paths[x_index]
#         path_y = self.paths[y_index]
#         # print(os.path.basename(path_x), os.path.basename(path_y))
#         x, x_seg = self.half_pair(pkload(path_x))
#         y, y_seg = self.half_pair(pkload(path_y))
#         x, y = x[None, ...], y[None, ...]
#         x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
#         x, x_seg = self.transforms([x, x_seg])
#         y, y_seg = self.transforms([y, y_seg])
#         x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
#         y = np.ascontiguousarray(y)
#         x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
#         y_seg = np.ascontiguousarray(y_seg)
#         x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
#         return x, y, x_seg, y_seg

#     def __len__(self):
#         return len(self.paths)*(len(self.paths)-1)