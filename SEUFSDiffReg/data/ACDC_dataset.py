from torch.utils.data import Dataset
import data.util_3D as Util
import os
import numpy as np
import scipy.io as sio
import json
import SimpleITK as sitk
import skimage as ski

class ACDCDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split+'.json')
        with open(datapath, 'r') as f:
            self.imageNum = json.load(f)

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        dataPath = self.imageNum[index]
        # data_ = sio.loadmat(dataPath)
        
        # end diastolic in cardiac MR images
        dataA = dataPath['image_ED']
        dataA = sitk.ReadImage(dataA)
        # print(f"sitk.ReadImage dataA size: {dataA.GetSize()}")  W(R-L) H(A-P) D(S-I)
        # sitk.ReadImage dataA size: (216, 256, 10)  
        # dataA = sitk.GetArrayFromImage(dataA).astype(np.float32).transpose(2, 1, 0)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)
        print(f"itk.GetArrayFromImage dataA shape: {dataA.shape}")
        dataA = dataA.transpose(2, 1, 0)
        print(f"** transpose dataA shape: {dataA.shape} max value {dataA.max()} min value {dataA.min()}")
        # print(dataA.shape)  [x,y,z]->[z,y,x]
        
        # end systolic in cardiac MR images
        dataB = dataPath['image_ES']
        dataB = sitk.ReadImage(dataB)
        dataB = sitk.GetArrayFromImage(dataB).astype(np.float32).transpose(2, 1, 0)
        print(f"** transpose dataB shape: {dataB.shape} max value {dataB.max()} min value {dataB.min()}")
        
        # red, blue and green represent the RV, LV, and MYO
        label_dataA = dataPath['label_ED']
        label_dataA = sitk.ReadImage(label_dataA)
        label_dataA = sitk.GetArrayFromImage(label_dataA).transpose(2, 1, 0)
        print(f"** transpose label_dataA shape: {label_dataA.shape} num: {np.unique(label_dataA)}")
        
        label_dataB = dataPath['label_ES']
        label_dataB = sitk.ReadImage(label_dataB)
        label_dataB = sitk.GetArrayFromImage(label_dataB).transpose(2, 1, 0)
        print(f"** transpose label_dataB shape: {label_dataB.shape} num: {np.unique(label_dataB)}")

        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nh, nw, nd = dataA.shape
        print(f"nh: {nh} nw: {nw} nd: {nd}")

        sh = int((nh - self.fineSize[0]) / 2)
        sw = int((nw - self.fineSize[1]) / 2)
        print(f"sh: {sh} sw: {sw}")
        dataA = dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        dataB = dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataA = label_dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataB = label_dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        print(f"After HW crop data shape: {dataA.shape}")

        if nd >= 32:
            sd = int((nd - self.fineSize[2]) / 2)
            dataA = dataA[..., sd:sd + self.fineSize[2]]
            dataB = dataB[..., sd:sd + self.fineSize[2]]
            label_dataA = label_dataA[..., sd:sd + self.fineSize[2]]
            label_dataB = label_dataB[..., sd:sd + self.fineSize[2]]
            print(f"After {nd} nd crop data shape: {dataA.shape}")
        else:
            sd = int((self.fineSize[2] - nd) / 2)
            dataA_ = np.zeros(self.fineSize)
            dataB_ = np.zeros(self.fineSize)
            dataA_[:, :, sd:sd + nd] = dataA
            dataB_[:, :, sd:sd + nd] = dataB
            label_dataA_ = np.zeros(self.fineSize)
            label_dataB_ = np.zeros(self.fineSize)
            label_dataA_[:, :, sd:sd + nd] = label_dataA
            label_dataB_[:, :, sd:sd + nd] = label_dataB
            dataA, dataB = dataA_, dataB_
            label_dataA, label_dataB = label_dataA_, label_dataB_
            print(f"After {nd} nd < 32 crop data shape: {dataA.shape}")
            
        # /home/liuhongzhi/Method/Registration/SEUFSDiffReg/data/util_3D.py  def transform_augment
        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))
        print(f"Moving image shape: {data.shape} Fixed image shape: {label.shape}")
        
        return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'Index': index}


class ACDCDataset_resize_name(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split+'.json')
        with open(datapath, 'r') as f:
            self.imageNum = json.load(f)

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        dataPath = self.imageNum[index]
        # data_ = sio.loadmat(dataPath)
        name = dataPath['image_ED'].split('/')[-2]
        # print(f"name: {name}")

        dataA = dataPath['image_ED']
        dataA = sitk.ReadImage(dataA)
        # ED_origin = dataA.GetOrigin()
        # ED_direction = dataA.GetDirection()
        # ED_spacing = dataA.GetSpacing()
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32).transpose(2,1,0)
        dataA = ski.transform.resize(dataA, self.fineSize, preserve_range=True)
        # print(dataA.shape)

        dataB = dataPath['image_ES']
        dataB = sitk.ReadImage(dataB)
        dataB = sitk.GetArrayFromImage(dataB).astype(np.float32).transpose(2,1,0)
        dataB = ski.transform.resize(dataB, self.fineSize, preserve_range=True)
        
        label_dataA = dataPath['label_ED']
        label_dataA = sitk.ReadImage(label_dataA)
        label_dataA = sitk.GetArrayFromImage(label_dataA).transpose(2,1,0)
        label_dataA = ski.transform.resize(label_dataA, self.fineSize, preserve_range=True)

        label_dataB = dataPath['label_ES']
        label_dataB = sitk.ReadImage(label_dataB)
        label_dataB = sitk.GetArrayFromImage(label_dataB).transpose(2,1,0)
        label_dataB = ski.transform.resize(label_dataB, self.fineSize, preserve_range=True)

        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        # nh, nw, nd = dataA.shape
        # print(dataA.shape,dataB.shape)

        # sh = int((nh - self.fineSize[0]) / 2)
        # sw = int((nw - self.fineSize[1]) / 2)
        # dataA = dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        # dataB = dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        # label_dataA = label_dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        # label_dataB = label_dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]

        # if nd >= 32:
        #     sd = int((nd - self.fineSize[2]) / 2)
        #     dataA = dataA[..., sd:sd + self.fineSize[2]]
        #     dataB = dataB[..., sd:sd + self.fineSize[2]]
        #     label_dataA = label_dataA[..., sd:sd + self.fineSize[2]]
        #     label_dataB = label_dataB[..., sd:sd + self.fineSize[2]]
        # else:
        #     sd = int((self.fineSize[2] - nd) / 2)
        #     dataA_ = np.zeros(self.fineSize)
        #     dataB_ = np.zeros(self.fineSize)
        #     dataA_[:, :, sd:sd + nd] = dataA
        #     dataB_[:, :, sd:sd + nd] = dataB
        #     label_dataA_ = np.zeros(self.fineSize)
        #     label_dataB_ = np.zeros(self.fineSize)
        #     label_dataA_[:, :, sd:sd + nd] = label_dataA
        #     label_dataB_[:, :, sd:sd + nd] = label_dataB
        #     dataA, dataB = dataA_, dataB_
        #     label_dataA, label_dataB = label_dataA_, label_dataB_
        
        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))
        
        # print(f"dataA.shape {dataA.shape} dataA.max {dataA.max()} dataA.min {dataA.min()}")
        # print(f"dataB.shape {dataB.shape} dataB.max {dataA.max()} dataB.min {dataA.min()}")
        # print(f"label_dataA.shape {label_dataA.shape} label_dataA.max {label_dataA.max()} label_dataA.min {label_dataA.min()}")
        # print(f"label_dataB.shape {label_dataB.shape} label_dataB.max {label_dataB.max()} label_dataB.min {label_dataB.min()}")
        
        # return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'Index': index, 
        #         'Name': name, 'Ori': ED_origin, 'Dir': ED_direction, 'Spa': ED_spacing}
        return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'Index': index, 
                'Name': name, 'Path': dataPath['image_ED'], 'imageED': dataA, 'imageES': dataB}
