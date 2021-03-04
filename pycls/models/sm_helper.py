import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class SMNorm(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.gamma = self.get_param(in_channels, constant=1.0)
        self.beta = self.get_param(in_channels, constant=0.0)
        self.kernel = self.get_kernel(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, channels, constant):
        param = torch.zeros([1, channels, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        nn.init.constant_(param, constant)
        return nn.Parameter(param)

    def get_kernel(self, in_channels, out_channels, kernel_size, groups):
        kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.18, 0.49, 1, 0.49, -0.18],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()

        kernel = kernel.repeat((out_channels, in_channels // groups, 1, 1))

        return kernel

    def forward(self, x):
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=(2, 2), groups=self.groups)
        x = self.gamma * x + self.beta
        return x


class CompareFixedSM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param = self.get_param(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.18, 0.49, 1, 0.49, -0.18],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        # kernel = torch.tensor([[-1/8, -1/8, -1/8],
        #                       [-1/8, 1, -1/8],
        #                       [-1/8, -1/8, -1/8]], requires_grad=False).cuda()
        kernel = kernel.repeat((out_channels, in_channels//groups, 1, 1))

        return kernel

    def forward(self, x):
        # x = F.conv2d(x, self.param, stride=self.stride, padding=self.padding, groups=self.groups)
        x = F.conv2d(x, self.param, stride=self.stride, padding=(2, 2), groups=self.groups)
        return x


class CompareFixedHP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param = self.get_param(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        # kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.18, 0.49, 1, 0.49, -0.18],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        kernel = torch.tensor([[-1/8, -1/8, -1/8],
                              [-1/8, 1, -1/8],
                              [-1/8, -1/8, -1/8]], requires_grad=False).cuda()
        # kernel = kernel/torch.sum(kernel)
        kernel = kernel.repeat((out_channels, in_channels//groups, 1, 1))

        return kernel

    def forward(self, x):
        # x = F.conv2d(x, self.param, stride=self.stride, padding=self.padding, groups=self.groups)
        x = F.conv2d(x, self.param, stride=self.stride, padding=(1, 1), groups=self.groups)
        return x



