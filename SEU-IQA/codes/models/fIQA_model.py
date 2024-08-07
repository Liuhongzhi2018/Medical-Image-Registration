import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class fIQA_Model(BaseModel):
    def __init__(self, opt):
        super(fIQA_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/networks.py  def define_IQA(opt)
        self.netIQA = networks.define_fIQA(opt).to(self.device)
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/networks.py  class BCERankingLoss(nn.Module)
        # self.rankLoss = networks.BCERankingLoss().to(self.device)
        self.noise_loss = nn.L1Loss().to(self.device)
        self.blur_loss = nn.L1Loss().to(self.device)
        self.color_loss = nn.L1Loss().to(self.device)
        self.contrast_loss = nn.L1Loss().to(self.device)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netIQA.train()

            # weight of loss
            self.l_weight = train_opt['loss_weight']  # loss_weight: !!float 1.0

            # optimizers
            optim_params = []
            wd_G = train_opt['weight_decay_IQA'] if train_opt['weight_decay_IQA'] else 0
            for k, v in self.netIQA.named_parameters():
                optim_params.append(v)
            # for k, v in self.rankLoss.named_parameters():
            #     optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, 
                                                lr=train_opt['lr_IQA'], 
                                                weight_decay=wd_G,
                                                betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_G)
            self.log_dict = OrderedDict()

    def feed_data(self, data, Train=True):
        if Train:
            self.Distortion = data['Dist_img'].to(self.device)
            self.scores = data['Dist_scores']
            self.scores[0] = self.scores[0].to(self.device)
            self.scores[1] = self.scores[1].to(self.device)
            self.scores[2] = self.scores[2].to(self.device)
            self.scores[3] = self.scores[3].to(self.device)
            self.tensor_one = torch.tensor([1]).float().to(self.device)
        else:
            self.Distortion = data['Dist_img'].to(self.device)

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        # Obtain Objective Score
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/networks.py def define_IQA(opt)
        self.noise_pre, self.blur_pre, self.color_pre, self.contrast_pre = self.netIQA(self.Distortion)
        self.noise_pre = self.noise_pre.float()
        self.blur_pre = self.blur_pre.float()
        self.color_pre = self.color_pre.float()
        self.contrast_pre = self.contrast_pre.float()
        self.loss_noise = self.noise_loss(self.noise_pre, self.scores[0])
        self.loss_blur = self.blur_loss(self.blur_pre, self.scores[1])
        self.loss_color = self.color_loss(self.color_pre, self.scores[2])
        self.loss_contrast = self.contrast_loss(self.contrast_pre, self.scores[3])
        
        self.loss = self.loss_noise + self.loss_blur + self.loss_color + self.loss_contrast
        
        # Predict perceptual judgment h from distance pair (d0, d1). See Paper of LPIPS.
        # B, _ = predict_pro_Ref_A.shape
        # var_judge = self.probability_AB.view(predict_pro_Ref_A.size())
        # self.loss = self.l_weight * (
        #     self.rankLoss.forward(predict_pro_Ref_A, predict_pro_Ref_B, var_judge * 2. - 1.)
        # )

        # Optimization
        self.clamp_weights()
        self.loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['loss_total'] = self.loss.item()
        self.log_dict['loss_noise'] = self.loss_noise.item()
        self.log_dict['loss_blur'] = self.loss_blur.item()
        self.log_dict['loss_color'] = self.loss_color.item()
        self.log_dict['loss_contrast'] = self.loss_contrast.item()

    def test(self):
        self.netIQA.eval()
        with torch.no_grad():
            self.noise_pre, self.blur_pre, self.color_pre, self.contrast_pre = self.netIQA(self.Distortion)
        self.netIQA.train()

    def clamp_weights(self):
        index = 0
        for module in self.netIQA.modules():
            try:
                if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                    index += 1
                    module.weight.data = torch.clamp(module.weight.data, min=0)
            except:
                pass

    def get_current_log(self):
        return self.log_dict

    def get_current_score(self):
        return self.noise_pre.detach().float().cpu(), self.blur_pre.detach().float().cpu(), \
               self.color_pre.detach().float().cpu(), self.contrast_pre.detach().float().cpu()

    def print_network(self):
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/base_model.py def get_network_description(self, network)
        s, n = self.get_network_description(self.netIQA)
        if isinstance(self.netIQA, nn.DataParallel) or isinstance(self.netIQA, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netIQA.__class__.__name__,
                                             self.netIQA.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netIQA.__class__.__name__)
        if self.rank <= 0:
            # logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info('IQA_Model Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            # logger.info(s)
            logger.info('IQA_Model Print_network: {}'.format(s))

    def load(self):
        try:
            load_path_G = self.opt['path']['pretrain_model_G']
            # load_path_R = self.opt['path']['pretrain_model_R']
        except:
            load_path_G, load_path_R = None, None

        # if (load_path_R is not None):
        #     logger.info('Loading model for R [{:s}] ...'.format(load_path_R))
        #     self.load_network(load_path_R, self.rankLoss, self.opt['path']['strict_load'])
        if (load_path_G is not None):
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netIQA, self.opt['path']['strict_load'])

    def save(self, iter_label):
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/base_model.py def save_network(self, network, network_label, iter_label)
        self.save_network(self.netIQA, 'G', iter_label)
        # self.save_network(self.rankLoss, 'R', iter_label)
