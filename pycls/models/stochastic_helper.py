import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class Stochastic(nn.Module):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channel, out_channel, prob):
        super(Stochastic).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prob = prob
        self.param = self.get_param(self.in_channels, self.out_channels, kernel_size=1, groups=1)

    def get_param(self, in_channels, out_channels, kernel_size=1, groups=1):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def subtract_feature(self, x, prob):
        """
        3x3 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        if prob > torch.rand(x.size()):
            x.size()    # B, C, H, W
        return weight

    def forward(self, x):
        weight = self.get_weight(self.param)
        if self.training:
            x = x
        return x


