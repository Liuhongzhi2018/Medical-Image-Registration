# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 22:21
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @File    : grid_sample_warp.py
# @Software: PyCharm

import torch
import torch.nn.functional as F


class warp3D:
    def __init__(self, setting='PyTorch Gridsample'):
        self.setting = setting

    def __call__(self, x, flow):
        return self._transform(x, flow)

    def _transform(self, x, flow):
        B, C, D, H, W = x.shape

        xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1)
        yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W)
        zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W)

        xx = xx.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        yy = yy.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        zz = zz.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)

        grid = torch.cat([xx, yy, zz], 1).float().cuda()

        dgrid = grid + flow

        dgrid[:, 0] = 2.0 * dgrid.clone()[:, 0] / max(W - 1, 1) - 1.0
        dgrid[:, 1] = 2.0 * dgrid.clone()[:, 1] / max(H - 1, 1) - 1.0
        dgrid[:, 2] = 2.0 * dgrid.clone()[:, 2] / max(D - 1, 1) - 1.0

        dgrid = dgrid.permute(0, 2, 3, 4, 1)
        output = F.grid_sample(x, dgrid, padding_mode='border')
        return output