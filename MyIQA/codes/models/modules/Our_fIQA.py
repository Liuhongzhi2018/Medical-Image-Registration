import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.parallel
import models.modules.module_util as module_util
from collections import OrderedDict

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), nn.ReLU(), ]
        self.model = nn.Sequential(*layers)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class Our_fIQA(nn.Module):
    def __init__(self, FENet='Alex', tune=False, control_bias=True, real_lpips=False, usecuda=False):
        super(Our_fIQA, self).__init__()

        # self.net = models.wide_resnet101_2(pretrained=True)
        self.net = models.resnext101_64x4d(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features, self.n_features)),
                                                  ('relu1', nn.ReLU()),
                                                  ('final', nn.Linear(self.n_features, 1))
                                                  ])
                                     )
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features, self.n_features)),
                                                  ('relu1', nn.ReLU()),
                                                  ('final', nn.Linear(self.n_features, 1))
                                                  ])
                                     )
        self.net.fc3 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features, self.n_features)),
                                                  ('relu1', nn.ReLU()),
                                                  ('final', nn.Linear(self.n_features, 1))
                                                  ])
                                     )
        self.net.fc4 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features, self.n_features)),
                                                  ('relu1', nn.ReLU()),
                                                  ('final', nn.Linear(self.n_features, 1))
                                                  ])
                                     )

    def forward(self, image_A):
        
        noise_head = self.net.fc1(self.net(image_A))
        blur_head = self.net.fc2(self.net(image_A))
        color_head = self.net.fc3(self.net(image_A))
        contrast_head = self.net.fc4(self.net(image_A))
        
        return noise_head, blur_head, color_head, contrast_head
