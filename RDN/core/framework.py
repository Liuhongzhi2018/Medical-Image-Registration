import numpy as np

import torch
import torch.nn as nn

from .utils import aug_transform, warp
from .networks import pynet

def augmentation(Img2):
    bs = Img2.shape[0]
    imgs = Img2.shape[2:5]  # D, H, W

    # control_fields = (aug_transform.sample_power(-0.4, 0.4, 3, [bs, 5, 5, 5, 3]) *
    #                   torch.Tensor(np.array(imgs).astype(np.float) // 4)).permute(0, 4, 1, 2, 3)
    control_fields = (aug_transform.sample_power(-0.4, 0.4, 3, [bs, 5, 5, 5, 3]) *
                      torch.Tensor(np.array(imgs).astype(np.float32) // 4)).permute(0, 4, 1, 2, 3)
    augFlow = (aug_transform.free_form_fields(imgs, control_fields)).cuda()  # B, C, D, H, W

    augImg2 = warp.warp3D()(Img2, augFlow)  # B, C, D, H, W

    return augImg2

class Framework_Teacher(nn.Module):
    def __init__(self, args, fixed=False):
        super(Framework_Teacher, self).__init__()
        self.args = args
        self.fixed = fixed
        self.flow_multiplier = 1
        self.reconstruction = warp.warp3D()
        self.defnet_0 = pynet.PyNet(flow_multiplier=self.flow_multiplier)
        if self.fixed == True:
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, Img1, augImg2):

        deforms_0 = self.defnet_0(Img1, augImg2)
        agg_flow_0 = deforms_0['flow']
        deforms = deforms_0



        return augImg2, augImg2, deforms, agg_flow_0
