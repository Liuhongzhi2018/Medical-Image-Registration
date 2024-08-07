import random
import numpy as np
import torch
import torch.utils.data as data
import data.data_util as util
import os
import torchvision.transforms as transforms

''' Train IQA on PIPAL dataset'''
class PIPALDataset(data.Dataset):
    def __init__(self, opt):
        super(PIPALDataset, self).__init__()
        # General Settings
        self.opt = opt
        self.crop_flag = self.opt['crop_flag'] # crop_flag: true
        self.crop_size = self.opt['crop_size'] # crop_size: 248
        self.norm_flag = self.opt['norm_flag'] # norm_flag : true
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # obtain basic roots of files
        self.mos_root = self.opt['mos_root']
        self.ref_root = self.opt['ref_root']
        self.dist_root = self.opt['dist_root']

        # obtain [distorted img A names] [distorted img B names] [ref image names] [MOS of A] [MOS of B]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL')
        self.ref_name, self.dist_A_name, self.dist_B_name, self.dist_A_scores, self.dist_B_scores = util.image_combinations(self.ref_root, 
                                                                                                                            self.dist_root, 
                                                                                                                            self.mos_root, 
                                                                                                                            phase='train')
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def all_img_paths(img_root)
        self.ref_paths = util.all_img_paths(self.ref_root)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py 
        self.dist_paths = util.all_img_paths(self.dist_root)

        assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):
        # get Reference Image
        reference_name = self.ref_name[index]
        reference_path = self.ref_paths[reference_name]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def read_img(path, size=None)
        img_ref = util.read_img(reference_path)
        # print(f"train reference_name {reference_name} {reference_path} img_ref shape: {img_ref.shape}")
        # train reference_name A0026.bmp /home/liuhongzhi/Data/PIPAL/training_part/Reference_train/A0026.bmp img_ref shape: (288, 288, 3)

        # get Distorted Image A
        distortion_A_name = self.dist_A_name[index]
        distortion_A_path = self.dist_paths[distortion_A_name]
        dist_A_ELO = self.dist_A_scores[index]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def read_img(path, size=None)
        img_dist_A = util.read_img(distortion_A_path)
        # print(f"train distortion_A_name {distortion_A_name} {distortion_A_path} img_dist_A shape: {img_dist_A.shape} dist_A_ELO: {dist_A_ELO}")
        #  train distortion_A_name A0026_01_06.bmp /home/liuhongzhi/Data/PIPAL/training_part/Distortion/Distortion_1/A0026_01_06.bmp img_dist_A shape: (288, 288, 3) dist_A_ELO: 1608.7916

        # get Distorted Image B
        distortion_B_name = self.dist_B_name[index]
        distortion_B_path = self.dist_paths[distortion_B_name]
        dist_B_ELO = self.dist_B_scores[index]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def read_img(path, size=None)
        img_dist_B = util.read_img(distortion_B_path)
        # print(f"train distortion_B_name {distortion_B_name} {distortion_B_path} img_dist_B shape: {img_dist_B.shape} dist_B_ELO: {dist_B_ELO}")
        # train distortion_B_name A0026_05_06.bmp /home/liuhongzhi/Data/PIPAL/training_part/Distortion/Distortion_1/A0026_05_06.bmp img_dist_B shape: (288, 288, 3) dist_B_ELO: 1454.9215

        # get the Probability that user prefers A than B
        probability_AB = torch.tensor(1 / (1 + np.power(10, (dist_B_ELO - dist_A_ELO) / 400))).float()
        # print(f"train probability_AB {probability_AB}")
        # train probability_AB 0.7080118060112

        # Choose whether crop image
        if self.crop_flag:
            H, W, C = img_ref.shape
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            img_ref = img_ref[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
            img_dist_A = img_dist_A[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
            img_dist_B = img_dist_B[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_ref.shape[2] == 3:
            img_ref = img_ref[:, :, [2, 1, 0]]
            img_dist_A = img_dist_A[:, :, [2, 1, 0]]
            img_dist_B = img_dist_B[:, :, [2, 1, 0]]
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()
        img_dist_A = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist_A, (2, 0, 1)))).float()
        img_dist_B = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist_B, (2, 0, 1)))).float()

        # Choose whether do Normalization
        if self.norm_flag:
            img_ref = self.normalize(img_ref)
            img_dist_A = self.normalize(img_dist_A)
            img_dist_B = self.normalize(img_dist_B)
            
        dataset_return = {'Ref': img_ref, 
                          'Dist_A': img_dist_A, 
                          'Dist_B': img_dist_B,
                          'probability_AB': probability_AB
                          }
        # print(f"train img_ref {img_ref.shape} img_dist_A: {img_dist_A.shape} img_dist_B: {img_dist_B.shape} probability_AB: {probability_AB}")
        # train img_ref torch.Size([3, 248, 248]) img_dist_A: torch.Size([3, 248, 248]) img_dist_B: torch.Size([3, 248, 248]) probability_AB: 0.7080118060112
        return dataset_return

    def __len__(self):
        return len(self.ref_name)
    
    
''' Train IQA on face IQA dataset'''
class fIQADataset(data.Dataset):
    def __init__(self, opt):
        super(fIQADataset, self).__init__()
        # General Settings
        self.opt = opt
        self.crop_flag = self.opt['crop_flag'] # crop_flag: true
        self.crop_size = self.opt['crop_size'] # crop_size: 224 and origin size is 224
        self.norm_flag = self.opt['norm_flag'] # norm_flag : true
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # obtain basic roots of files
        self.mos_root = self.opt['mos_root']
        # self.ref_root = self.opt['ref_root']
        self.dist_root = self.opt['dist_root']

        # obtain [distorted img A names] [distorted img B names] [ref image names] [MOS of A] [MOS of B]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL')
        self.dist_name, self.dist_scores = util.image_combinations_fIQA(self.dist_root, 
                                                                        self.mos_root, 
                                                                        phase='train')
        # print(f"Name: {self.dist_name} \nScores: {self.dist_scores}")
        # print(f"Number of images: {len(self.dist_name)} \nNumber of scores: {len(self.dist_scores)}")
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def all_img_paths(img_root)
        # self.ref_paths = util.all_img_paths(self.ref_root)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py 
        self.dist_paths = util.all_img_paths(self.dist_root)

        # assert self.ref_paths, 'Error: ref path is empty.'
        assert self.dist_paths, 'Error: distortion path is empty.'

    def __getitem__(self, index):

        # get Distorted Image A
        distortion_name = os.path.basename(self.dist_name[index])
        distortion_path = self.dist_name[index]
        dist_scores = self.dist_scores[index]
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/data_util.py def read_img(path, size=None)
        img_dist = util.read_img(distortion_path)
        # print(f"train distortion_A_name {distortion_A_name} {distortion_A_path} img_dist_A shape: {img_dist_A.shape} dist_A_ELO: {dist_A_ELO}")
        #  train distortion_A_name A0026_01_06.bmp /home/liuhongzhi/Data/PIPAL/training_part/Distortion/Distortion_1/A0026_01_06.bmp img_dist_A shape: (288, 288, 3) dist_A_ELO: 1608.7916

        # Choose whether crop image
        if self.crop_flag:
            H, W, C = img_dist.shape
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            img_dist = img_dist[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_dist.shape[2] == 3:
            img_dist = img_dist[:, :, [2, 1, 0]]
        img_dist = torch.from_numpy(np.ascontiguousarray(np.transpose(img_dist, (2, 0, 1)))).float()

        # Choose whether do Normalization
        if self.norm_flag:
            img_dist = self.normalize(img_dist)
            
        dataset_return = {'name':distortion_name,
                          'Dist_img': img_dist, 
                          'Dist_scores': dist_scores
                          }
        # print(f"dataset_return: {dataset_return}")
        # print(f"train img_ref {img_ref.shape} img_dist_A: {img_dist_A.shape} img_dist_B: {img_dist_B.shape} probability_AB: {probability_AB}")
        # train img_ref torch.Size([3, 248, 248]) img_dist_A: torch.Size([3, 248, 248]) img_dist_B: torch.Size([3, 248, 248]) probability_AB: 0.7080118060112
        return dataset_return

    def __len__(self):
        return len(self.dist_name)
    

''' Train IQA on BAPPS dataset'''
class BAPPSDataset(data.Dataset):
    def __init__(self, opt):
        super(BAPPSDataset, self).__init__()

        # General Settings
        self.opt = opt
        self.train_valid = self.opt['train_valid']
        self.crop_flag = self.opt['crop_flag']
        self.crop_size = self.opt['crop_size']
        self.norm_flag = self.opt['norm_flag']
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.bapps_train_root = self.opt['train_root']
        self.bapps_valid_root = self.opt['valid_root']

        # Choose which partition. Obtain paths.
        if self.train_valid == 'both':
            self.roots = [self.bapps_train_root + '/traditional', self.bapps_train_root + '/mix',
                          self.bapps_train_root + '/cnn',
                          self.bapps_valid_root + '/cnn', self.bapps_valid_root + '/color',
                          self.bapps_valid_root + '/deblur',
                          self.bapps_valid_root + '/frameinterp', self.bapps_valid_root + '/superres',
                          self.bapps_valid_root + '/traditional'
                          ]
        elif self.train_valid == 'train':
            self.roots = [self.bapps_train_root + '/traditional', self.bapps_train_root + '/mix',
                          self.bapps_train_root + '/cnn']
        elif self.train_valid == 'valid':
            self.roots = [self.bapps_valid_root + '/cnn', self.bapps_valid_root + '/color',
                          self.bapps_valid_root + '/deblur',
                          self.bapps_valid_root + '/frameinterp', self.bapps_valid_root + '/superres',
                          self.bapps_valid_root + '/traditional'
                          ]
        else:
            assert 'Please check configuration file. Setting of BAPPS on train_valid get something wrong. '

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.twoafc_ref_paths = util.make_dataset(self.dir_ref)
        self.twoafc_ref_paths = sorted(self.twoafc_ref_paths)

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = util.make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = util.make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.judge_paths = util.make_dataset(self.dir_J, mode='np')
        self.judge_paths = sorted(self.judge_paths)

        assert self.twoafc_ref_paths, 'Error: 2afc ref path is empty.'
        assert self.p0_paths, 'Error: 2afc p0 path is empty.'
        assert self.p1_paths, 'Error: 2afc p1 path is empty.'
        assert self.judge_paths, 'Error: 2afc judge path is empty.'

    def __getitem__(self, index):
        # get Distorted image A
        twoafc_p0_path = self.p0_paths[index]
        p0_img = util.read_img(twoafc_p0_path)

        # get Distorted image B
        twoafc_p1_path = self.p1_paths[index]
        p1_img = util.read_img(twoafc_p1_path)

        # get Distorted image Ref
        twoafc_ref_path = self.twoafc_ref_paths[index]
        twoafc_ref_img = util.read_img(twoafc_ref_path)

        # get the probability that user prefers A than B
        twoafc_judge_path = self.judge_paths[index]
        probability_AB = torch.tensor(float(np.load(twoafc_judge_path))).float()  # [0,1]

        # Choose whether to crop image
        if self.crop_flag:
            H, W, C = twoafc_ref_img.shape
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            twoafc_ref_img = twoafc_ref_img[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
            p0_img = p0_img[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]
            p1_img = p1_img[rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if twoafc_ref_img.shape[2] == 3:
            twoafc_ref_img = twoafc_ref_img[:, :, [2, 1, 0]]
            p0_img = p0_img[:, :, [2, 1, 0]]
            p1_img = p1_img[:, :, [2, 1, 0]]
        twoafc_ref_img = torch.from_numpy(np.ascontiguousarray(np.transpose(twoafc_ref_img, (2, 0, 1)))).float()
        p0_img = torch.from_numpy(np.ascontiguousarray(np.transpose(p0_img, (2, 0, 1)))).float()
        p1_img = torch.from_numpy(np.ascontiguousarray(np.transpose(p1_img, (2, 0, 1)))).float()

        # Choose whether to do Normalization
        if self.norm_flag:
            twoafc_ref_img = self.normalize(twoafc_ref_img)
            p0_img = self.normalize(p0_img)
            p1_img = self.normalize(p1_img)

        dataset_return = {
            'Ref': twoafc_ref_img, 'Dist_A': p0_img, 'Dist_B': p1_img,
            'probability_AB': probability_AB
        }
        return dataset_return

    def __len__(self):
        return len(self.twoafc_ref_paths)
