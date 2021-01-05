import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class BlurKernelConv(nn.Conv2d):

    """
    blurred kernel convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.replication_pad = nn.ReplicationPad2d(1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.filt = self.get_filt(self.in_channels, self.out_channels, groups=self.in_channels)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        return nn.Parameter(param)

    def get_filt(self, in_channels, out_channels, groups):
        filt = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]).cuda()
        filt = filt/torch.sum(filt)
        filt = filt.repeat((in_channels, in_channels//groups, 1, 1))
        return filt


    def get_weight(self, param, groups):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        # param = self.replication_pad(param)
        weight = F.conv2d(param, self.filt, bias=None, stride=1, padding=1, dilation=1, groups=groups)

        return weight

    def forward(self, x):
        weight = self.param
        weight = self.get_weight(weight, self.in_channels)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)

        return x