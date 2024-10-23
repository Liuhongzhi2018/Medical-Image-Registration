import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import warp


class Estimator(nn.Module):
    def __init__(self, inputc):
        super(Estimator, self).__init__()

        self.conv1 = nn.Conv3d(inputc, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(8, 3, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        flow = self.conv4(x)
        return flow


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(16, 24, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(24, 32, kernel_size=3, stride=2, padding=1)

        # Parameters are less compared with transposed conv
        self.conv3_2 = nn.Conv3d(32, 24, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv3d(24 + 24, 24, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv3d(24, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv3d(16 + 16, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv3d(8 + 8, 8, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(8 + 1, 8, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_2x = self.lrelu(self.conv1(x))
        x_4x = self.lrelu(self.conv2(x_2x))
        x_8x = self.lrelu(self.conv3(x_4x))
        x_16x = self.lrelu(self.conv4(x_8x))

        _, _, d, h, w = x_8x.shape
        x_16x_up = self.lrelu(self.conv3_2(F.interpolate(x_16x, size=(d, h, w), mode='trilinear')))
        x_8x_f = self.lrelu(self.conv3_3(torch.cat([x_16x_up, x_8x], 1)))

        _, _, d, h, w = x_4x.shape
        x_8x_up = self.lrelu(self.conv2_2(F.interpolate(x_8x_f, size=(d, h, w), mode='trilinear')))
        x_4x_f = self.lrelu(self.conv2_3(torch.cat([x_8x_up, x_4x], 1)))

        _, _, d, h, w = x_2x.shape
        x_4x_up = self.lrelu(self.conv1_2(F.interpolate(x_4x_f, size=(d, h, w), mode='trilinear')))
        x_2x_f = self.lrelu(self.conv1_3(torch.cat([x_4x_up, x_2x], 1)))

        return {'1/2': x_2x_f, '1/4': x_4x_f, '1/8': x_8x_f, '1/16': x_16x}


class PyNet(nn.Module):
    def __init__(self, flow_multiplier=1.):
        super(PyNet, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.global_iters = 4

        self.extraction = PSPNet()

        self.estimator_16x_1st = nn.ModuleList([Estimator(inputc=(64 + 3)) for _ in range(self.global_iters)])
        self.estimator_8x_1st = nn.ModuleList([Estimator(inputc=(48 + 3)) for _ in range(self.global_iters)])
        self.estimator_4x_1st = nn.ModuleList([Estimator(inputc=(32 + 3)) for _ in range(self.global_iters)])
        self.estimator_2x_1st = nn.ModuleList([Estimator(inputc=(16 + 3)) for _ in range(self.global_iters)])

        self.estimator_16x_2nd = nn.ModuleList([Estimator(inputc=(64 + 3)) for _ in range(self.global_iters)])
        self.estimator_8x_2nd = nn.ModuleList([Estimator(inputc=(48 + 3)) for _ in range(self.global_iters)])
        self.estimator_4x_2nd = nn.ModuleList([Estimator(inputc=(32 + 3)) for _ in range(self.global_iters)])
        self.estimator_2x_2nd = nn.ModuleList([Estimator(inputc=(16 + 3)) for _ in range(self.global_iters)])

        self.estimator_16x_3rd = nn.ModuleList([Estimator(inputc=(64 + 3)) for _ in range(self.global_iters)])
        self.estimator_8x_3rd = nn.ModuleList([Estimator(inputc=(48 + 3)) for _ in range(self.global_iters)])
        self.estimator_4x_3rd = nn.ModuleList([Estimator(inputc=(32 + 3)) for _ in range(self.global_iters)])
        self.estimator_2x_3rd = nn.ModuleList([Estimator(inputc=(16 + 3)) for _ in range(self.global_iters)])

        self.estimator_16x_4th = nn.ModuleList([Estimator(inputc=(64 + 3)) for _ in range(self.global_iters)])
        self.estimator_8x_4th = nn.ModuleList([Estimator(inputc=(48 + 3)) for _ in range(self.global_iters)])
        self.estimator_4x_4th = nn.ModuleList([Estimator(inputc=(32 + 3)) for _ in range(self.global_iters)])
        self.estimator_2x_4th = nn.ModuleList([Estimator(inputc=(16 + 3)) for _ in range(self.global_iters)])

        self.reconstruction = warp.warp3D()

    def forward(self, image1, image2):
        f_fea = self.extraction(image1)
        m_fea = self.extraction(image2)
        b, c, d_16x, h_16x, w_16x = f_fea['1/16'].shape
        b, c, d_8x, h_8x, w_8x = f_fea['1/8'].shape
        b, c, d_4x, h_4x, w_4x = f_fea['1/4'].shape
        b, c, d_2x, h_2x, w_2x = f_fea['1/2'].shape
        b, c, d_1x, h_1x, w_1x = image1.shape

        deform = []

        for i in range(self.global_iters):
            if i == 0:
                m_fea_16x = m_fea['1/16']
                m_fea_8x = m_fea['1/8']
                m_fea_4x = m_fea['1/4']
                m_fea_2x = m_fea['1/2']
            else:
                flow_16x_pre = F.interpolate(flow_pre, size=(d_16x, h_16x, w_16x), mode='trilinear') / 8.0
                flow_8x_pre = F.interpolate(flow_pre, size=(d_8x, h_8x, w_8x), mode='trilinear') / 4.0
                flow_4x_pre = F.interpolate(flow_pre, size=(d_4x, h_4x, w_4x), mode='trilinear') / 2.0
                flow_2x_pre = flow_pre

                m_fea_16x = self.reconstruction(m_fea['1/16'], flow_16x_pre)
                m_fea_8x = self.reconstruction(m_fea['1/8'], flow_8x_pre)
                m_fea_4x = self.reconstruction(m_fea['1/4'], flow_4x_pre)
                m_fea_2x = self.reconstruction(m_fea['1/2'], flow_2x_pre)

            # 1/16
            flow_32x_up = torch.zeros(size=(b, 3, d_16x, h_16x, w_16x)).cuda()
            fm_16x_1st = (torch.cat([f_fea['1/16'], m_fea_16x], 1))
            flow_16x_1st = self.estimator_16x_1st[i](torch.cat([fm_16x_1st, flow_32x_up], 1))

            flow_16x_1st = self.reconstruction(flow_32x_up, flow_16x_1st) + flow_16x_1st

            m_fea_16x_warped_2nd = self.reconstruction(m_fea_16x, flow_16x_1st)
            fm_16x_2nd = (torch.cat([f_fea['1/16'], m_fea_16x_warped_2nd], 1))
            flow_16x_2nd = self.estimator_16x_2nd[i](torch.cat([fm_16x_2nd, flow_16x_1st], 1))

            flow_16x_2nd = self.reconstruction(flow_16x_1st, flow_16x_2nd) + flow_16x_2nd

            m_fea_16x_warped_3rd = self.reconstruction(m_fea_16x, flow_16x_2nd)
            fm_16x_3rd = (torch.cat([f_fea['1/16'], m_fea_16x_warped_3rd], 1))
            flow_16x_3rd = self.estimator_16x_3rd[i](torch.cat([fm_16x_3rd, flow_16x_2nd], 1))

            flow_16x_3rd = self.reconstruction(flow_16x_2nd, flow_16x_3rd) + flow_16x_3rd

            m_fea_16x_warped_4th = self.reconstruction(m_fea_16x, flow_16x_3rd)
            fm_16x_4th = (torch.cat([f_fea['1/16'], m_fea_16x_warped_4th], 1))
            flow_16x_4th = self.estimator_16x_4th[i](torch.cat([fm_16x_4th, flow_16x_3rd], 1))

            flow_16x = self.reconstruction(flow_16x_3rd, flow_16x_4th) + flow_16x_4th

            # 1/8
            flow_16x_up = F.interpolate(flow_16x, size=(d_8x, h_8x, w_8x), mode='trilinear') * 2.0
            m_fea_8x_warped_1st = self.reconstruction(m_fea_8x, flow_16x_up)
            fm_8x_1st = (torch.cat([f_fea['1/8'], m_fea_8x_warped_1st], 1))
            flow_8x_1st = self.estimator_8x_1st[i](torch.cat([fm_8x_1st, flow_16x_up], 1))

            flow_8x_1st = self.reconstruction(flow_16x_up, flow_8x_1st) + flow_8x_1st

            m_fea_8x_warped_2nd = self.reconstruction(m_fea_8x, flow_8x_1st)
            fm_8x_2nd = (torch.cat([f_fea['1/8'], m_fea_8x_warped_2nd], 1))
            flow_8x_2nd = self.estimator_8x_2nd[i](torch.cat([fm_8x_2nd, flow_8x_1st], 1))

            flow_8x_2nd = self.reconstruction(flow_8x_1st, flow_8x_2nd) + flow_8x_2nd

            m_fea_8x_warped_3rd = self.reconstruction(m_fea_8x, flow_8x_2nd)
            fm_8x_3rd = (torch.cat([f_fea['1/8'], m_fea_8x_warped_3rd], 1))
            flow_8x_3rd = self.estimator_8x_3rd[i](torch.cat([fm_8x_3rd, flow_8x_2nd], 1))

            flow_8x_3rd = self.reconstruction(flow_8x_2nd, flow_8x_3rd) + flow_8x_3rd

            m_fea_8x_warped_4th = self.reconstruction(m_fea_8x, flow_8x_3rd)
            fm_8x_4th = (torch.cat([f_fea['1/8'], m_fea_8x_warped_4th], 1))
            flow_8x_4th = self.estimator_8x_4th[i](torch.cat([fm_8x_4th, flow_8x_3rd], 1))

            flow_8x = self.reconstruction(flow_8x_3rd, flow_8x_4th) + flow_8x_4th

            # 1/4
            flow_8x_up = F.interpolate(flow_8x, size=(d_4x, h_4x, w_4x), mode='trilinear') * 2.0
            m_fea_4x_warped_1st = self.reconstruction(m_fea_4x, flow_8x_up)
            fm_4x_1st = (torch.cat([f_fea['1/4'], m_fea_4x_warped_1st], 1))
            flow_4x_1st = self.estimator_4x_1st[i](torch.cat([fm_4x_1st, flow_8x_up], 1))

            flow_4x_1st = self.reconstruction(flow_8x_up, flow_4x_1st) + flow_4x_1st

            m_fea_4x_warped_2nd = self.reconstruction(m_fea_4x, flow_4x_1st)
            fm_4x_2nd = (torch.cat([f_fea['1/4'], m_fea_4x_warped_2nd], 1))
            flow_4x_2nd = self.estimator_4x_2nd[i](torch.cat([fm_4x_2nd, flow_4x_1st], 1))

            flow_4x_2nd = self.reconstruction(flow_4x_1st, flow_4x_2nd) + flow_4x_2nd

            m_fea_4x_warped_3rd = self.reconstruction(m_fea_4x, flow_4x_2nd)
            fm_4x_3rd = (torch.cat([f_fea['1/4'], m_fea_4x_warped_3rd], 1))
            flow_4x_3rd = self.estimator_4x_3rd[i](torch.cat([fm_4x_3rd, flow_4x_2nd], 1))

            flow_4x_3rd = self.reconstruction(flow_4x_2nd, flow_4x_3rd) + flow_4x_3rd

            m_fea_4x_warped_4th = self.reconstruction(m_fea_4x, flow_4x_3rd)
            fm_4x_4th = (torch.cat([f_fea['1/4'], m_fea_4x_warped_4th], 1))
            flow_4x_4th = self.estimator_4x_4th[i](torch.cat([fm_4x_4th, flow_4x_3rd], 1))

            flow_4x = self.reconstruction(flow_4x_3rd, flow_4x_4th) + flow_4x_4th

            # 1/2
            flow_4x_up = F.interpolate(flow_4x, size=(d_2x, h_2x, w_2x), mode='trilinear') * 2.0
            m_fea_2x_warped_1st = self.reconstruction(m_fea_2x, flow_4x_up)
            fm_2x_1st = (torch.cat([f_fea['1/2'], m_fea_2x_warped_1st], 1))
            flow_2x_1st = self.estimator_2x_1st[i](torch.cat([fm_2x_1st, flow_4x_up], 1))

            flow_2x_1st = self.reconstruction(flow_4x_up, flow_2x_1st) + flow_2x_1st

            m_fea_2x_warped_2nd = self.reconstruction(m_fea_2x, flow_2x_1st)
            fm_2x_2nd = (torch.cat([f_fea['1/2'], m_fea_2x_warped_2nd], 1))
            flow_2x_2nd = self.estimator_2x_2nd[i](torch.cat([fm_2x_2nd, flow_2x_1st], 1))

            flow_2x_2nd = self.reconstruction(flow_2x_1st, flow_2x_2nd) + flow_2x_2nd

            m_fea_2x_warped_3rd = self.reconstruction(m_fea_2x, flow_2x_2nd)
            fm_2x_3rd = (torch.cat([f_fea['1/2'], m_fea_2x_warped_3rd], 1))
            flow_2x_3rd = self.estimator_2x_3rd[i](torch.cat([fm_2x_3rd, flow_2x_2nd], 1))

            flow_2x_3rd = self.reconstruction(flow_2x_2nd, flow_2x_3rd) + flow_2x_3rd\

            m_fea_2x_warped_4th = self.reconstruction(m_fea_2x, flow_2x_3rd)
            fm_2x_4th = (torch.cat([f_fea['1/2'], m_fea_2x_warped_4th], 1))
            flow_2x_4th = self.estimator_2x_4th[i](torch.cat([fm_2x_4th, flow_2x_3rd], 1))

            flow_2x = self.reconstruction(flow_2x_3rd, flow_2x_4th) + flow_2x_4th

            # final flow
            flow = F.interpolate(flow_2x, size=(d_1x, h_1x, w_1x), mode='trilinear') * 2.0
            deform.append(flow * self.flow_multiplier)

            if i == 0:
                flow_pre = flow_2x
            else:
                flow_pre = self.reconstruction(flow_pre, flow_2x) + flow_2x

        flow_final = F.interpolate(flow_pre, size=(d_1x, h_1x, w_1x), mode='trilinear') * 2.0 * self.flow_multiplier

        return {'flow': flow_final, 'deform': deform}
