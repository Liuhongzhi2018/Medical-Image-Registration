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
        self.opt = opt
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.dataset_name = self.opt['name']
        self.ref_root = self.opt['ref_root']
        self.dist_root = self.opt['dist_root']

        self.ref_names, self.dist_names = util.image_combinations(self.ref_root, 
                                                                  self.dist_root, 
                                                                  phase='test',
                                                                  mos_root=None, 
                                                                  dataset_name=self.dataset_name)
        self.ref_paths = util.all_img_paths(self.ref_root)
        self.dist_paths = util.all_img_paths(self.dist_root)
        assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):

        reference_name = self.ref_names[index]
        reference_path = self.ref_paths[reference_name]
        ref = util.read_img(reference_path)

        distortion_name = self.dist_names[index]
        distortion_path = self.dist_paths[distortion_name]
        dist = util.read_img(distortion_path)

        if ref.shape[2] == 3:
            img_ref = ref[:, :, [2, 1, 0]]
            img_dist = dist[:, :, [2, 1, 0]]
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()
        img_dist = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist, (2, 0, 1)))).float()

        img_ref = self.normalize(img_ref)
        img_dist = self.normalize(img_dist)

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
        self.opt = opt
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.dist_root = self.opt['dist_root']

        self.dist_names = util.image_combinations_fIQA(self.dist_root, 
                                                       mos_root=None, 
                                                       phase='test')
        self.dist_paths = util.all_img_paths(self.dist_root)
        # assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):

        distortion_name = os.path.basename(self.dist_names[index])
        distortion_path = self.dist_names[index]
        dist = util.read_img(distortion_path)

        if dist.shape[2] == 3:
            img_dist = dist[:, :, [2, 1, 0]]
        img_dist = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist, (2, 0, 1)))).float()

        img_dist = self.normalize(img_dist)

        return {'name': distortion_name, 
                'Dist_img': img_dist
                }

    def __len__(self):
        return len(self.dist_paths)

