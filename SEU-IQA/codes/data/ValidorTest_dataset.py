import os
import numpy as np
import torch
import torch.utils.data as data
import data.data_util as util
import torchvision.transforms as transforms

class ValidDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(ValidDataset, self).__init__()
        # General Setting
        self.opt = opt
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # obtain basic roots of files
        self.dataset_name = self.opt['name']
        self.ref_root = self.opt['ref_root']
        self.dist_root = self.opt['dist_root']

        # obtain [ref image names], [distortion image names]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL')
        self.ref_names, self.dist_names = util.image_combinations(self.ref_root, 
                                                                  self.dist_root, 
                                                                  phase='test',
                                                                  mos_root=None, 
                                                                  dataset_name=self.dataset_name)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def all_img_paths(img_root)
        self.ref_paths = util.all_img_paths(self.ref_root)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py
        self.dist_paths = util.all_img_paths(self.dist_root)
        assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):

        # get Ref image
        reference_name = self.ref_names[index]
        reference_path = self.ref_paths[reference_name]
        ref = util.read_img(reference_path)
        # print(f"valid reference_name {reference_name} {reference_path} ref shape: {ref.shape}")
        # valid reference_name A0185.bmp /home/liuhongzhi/Data/PIPAL/validation_part/Reference_valid/A0185.bmp ref shape: (288, 288, 3)

        # get distortion A image
        distortion_name = self.dist_names[index]
        distortion_path = self.dist_paths[distortion_name]
        dist = util.read_img(distortion_path)
        # print(f"valid distortion_name {distortion_name} {distortion_path} dist shape: {dist.shape}")
        # valid distortion_name A0185_00_06.bmp /home/liuhongzhi/Data/PIPAL/validation_part/Distortion_valid/A0185_00_06.bmp dist shape: (288, 288, 3)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if ref.shape[2] == 3:
            img_ref = ref[:, :, [2, 1, 0]]
            img_dist = dist[:, :, [2, 1, 0]]
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()
        img_dist = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist, (2, 0, 1)))).float()

        # Choose whether do Normalization
        img_ref = self.normalize(img_ref)
        img_dist = self.normalize(img_dist)
        # print(f"valid img_ref: {img_ref.shape} img_dist: {img_dist.shape}")
        # alid img_ref: torch.Size([3, 288, 288]) img_dist: torch.Size([3, 288, 288])

        return {'Ref': img_ref, 
                'Distortion': img_dist, 
                'Dis_Name': distortion_name
                }

    def __len__(self):
        return len(self.dist_paths)
    
    
    
class ValidfIQAdataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(ValidfIQAdataset, self).__init__()
        # General Setting
        self.opt = opt
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # obtain basic roots of files
        self.dist_root = self.opt['dist_root']

        # obtain [ref image names], [distortion image names]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL')
        self.dist_names = util.image_combinations_fIQA(self.dist_root, 
                                                       mos_root=None, 
                                                       phase='test')
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def all_img_paths(img_root)
        # self.ref_paths = util.all_img_paths(self.ref_root)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py
        self.dist_paths = util.all_img_paths(self.dist_root)
        # assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):

        # get distortion image  
        distortion_name = os.path.basename(self.dist_names[index])
        distortion_path = self.dist_names[index]
        dist = util.read_img(distortion_path)
        # print(f"valid distortion_name {distortion_name} {distortion_path} dist shape: {dist.shape}")
        # valid distortion_name A0185_00_06.bmp /home/liuhongzhi/Data/PIPAL/validation_part/Distortion_valid/A0185_00_06.bmp dist shape: (288, 288, 3)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if dist.shape[2] == 3:
            img_dist = dist[:, :, [2, 1, 0]]
        img_dist = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist, (2, 0, 1)))).float()

        # Choose whether do Normalization
        img_dist = self.normalize(img_dist)
        # print(f"valid img_ref: {img_ref.shape} img_dist: {img_dist.shape}")
        # alid img_ref: torch.Size([3, 288, 288]) img_dist: torch.Size([3, 288, 288])

        return {'name': distortion_name, 
                'Dist_img': img_dist
                }

    def __len__(self):
        return len(self.dist_paths)

