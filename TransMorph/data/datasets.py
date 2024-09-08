import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


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
        f_name = f_paths[0].split("/")[-1]
        x, x_seg, y, y_seg = f_paths[0], f_paths[1], f_paths[2], f_paths[3]

        x = self.read_vol(x)
        # x_seg = self.read_vol(x_seg)
        y = self.read_vol(y)
        # y_seg = self.read_vol(y_seg)
        # print(f"LPBADataset x shape: {x.shape} y shape: {y.shape}")
        # LPBADataset x shape: (160, 192, 160) y shape: (160, 192, 160)
        # print(f"LPBADataset x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # LPBADataset x seg shape: [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43 44  45  46  47  48  49  50  61  62  
        # 63  64  65  66  67  68  81  82  83 84  85  86  87  88  89  90  91  92 101 102 121 122 161 162 163 164 165 166 181 182] 
        # y seg shape: [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43 44  45 46  47  48  
        # 49  50  61  62  63  64  65  66  67  68  81  82  83 84  85  86  87  88  89  90  91  92 101 102 121 122 161 162 163 164 165 166 181 182]

        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        
        # [Bsize, channels, Height, Width, Depth]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(f"OASISDataset out x shape: {x.shape} y shape: {y.shape}")
        # OASISDataset out x shape: torch.Size([1, 160, 224, 192]) y shape: torch.Size([1, 160, 224, 192])
        # return f_name, x, y
        return x, y

    def __len__(self):
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
        # print(f"LPBADatasetVal x shape: {x.shape} y shape: {y.shape}")
        # LPBADatasetVal x shape: (160, 192, 160) y shape: (160, 192, 160)
        # print(f"LPBADatasetVal x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # LPBADatasetVal x seg shape: [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43 
        # 44  45  46  47  48  49  50  61  62  63  64  65  66  67  68  81  82  83 84  85  86  87  88  89  90  91  
        # 92 101 102 121 122 161 162 163 164 165 166 181 182] 
        # y seg shape: [  0  21  22  23  24  25  26  27  28  29  30  31  32  33  34  41  42  43 44  45  46  47  48  49  
        # 50  61  62  63  64  65  66  67  68  81  82  83 84  85  86  87  88  89  90  91  92 101 102 121 122 161 162 163 164 165 166 181 182]

        x_seg = self.translabel(x_seg)
        y_seg = self.translabel(y_seg)
        # print(f"LPBADatasetVal translabel {f_name} x_seg: {np.unique(x_seg)} y_seg : {np.unique(y_seg)}")
        # LPBADatasetVal translabel S10.delineation.skullstripped.nii.gz x_seg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56] 
        # y_seg : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
        # 48 49 50 51 52 53 54 55 56]

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
        # OASISDataset x shape: (160, 384, 384) y shape: (160, 384, 384)
        # print(f"OASISDataset x seg shape: {np.unique(x_seg)} y seg shape: {np.unique(y_seg)}")
        # OASISDataset x seg shape: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
        #                            21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
        # OASISDataset x seg shape: [0 1 2 3 4 5] y seg shape: [0 1 2 3 4 5]
        
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
    

class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)