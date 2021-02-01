# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch
import torch.nn.parallel
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class DogBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        # kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.18, 0.49, 1, 0.49, -0.18],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
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
        x = self.reflection_pad(x)
        x = F.conv2d(x, self.param, stride=self.stride, groups=self.groups)
        return x


class CustomBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect', sigma=0.8, kernel_norm=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.kernel_norm = kernel_norm
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.groups, sigma=sigma)
        # self.kernel = self.get_weight(self.param)

    def get_param(self, in_channels, out_channels, groups, sigma=.8):
        param = torch.ones([out_channels, in_channels // groups, 1], dtype=torch.float,
                           requires_grad=False)
        param = param.cuda()
        param *= sigma
        return nn.Parameter(param)

    def get_weight(self, sigma):

        x = self.get_gaussian(sigma, loc=0)
        y = self.get_gaussian(sigma, loc=1)
        if self.kernel_size == 3:
            param = torch.cat([y, x, y], dim=2)
        elif self.kernel_size == 5:
            z = self.get_gaussian(sigma, loc=4)
            param = torch.cat([z, y, x, y, z], dim=2)
        kernel = torch.einsum('bci,bcj->bcij', param, param)

        # kernel = torch.tensor([[0.04, 0.04, 0.04, 0.04, 0.04],
        #                        [0.04, 0.04, 0.04, 0.04, 0.04],
        #                        [0.04, 0.04, 0.04, 0.04, 0.04],
        #                        [0.04, 0.04, 0.04, 0.04, 0.04],
        #                        [0.04, 0.04, 0.04, 0.04, 0.04]], requires_grad=False).cuda()
        if self.kernel_norm == True:
            kernel_sum = kernel.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
            kernel = kernel / kernel_sum

        return kernel

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def forward(self, x):
        kernel = self.get_weight(self.param)
        x = self.reflection_pad(x)
        x = F.conv2d(x, kernel, stride=self.stride, groups=self.groups)
        return x


class SortBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.20, mul=1.0)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.25, mul=1.0)
        self.param3 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.01, mul=1.0)
        self.param4 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.01, mul=1.0)
        self.param5 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.01, mul=1.0)

    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375, mul=1):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        param.data.normal_(mean=0, std=np.sqrt(2.0 / fan_out))   # 0.2, 0.05
        # param.data.uniform_(0, np.sqrt(6.0 / fan_out))
        param *= mul
        # nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight(self, param1, param2, param3):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2))
        param3 = (F.relu(param3) + F.relu(param3))
        param = torch.cat([param1, param2, param3], dim=2)
        param_descend, _ = torch.sort(param, dim=2, descending=True)
        # param_descend[:, :, 0] = param_descend[:, :, 0] * 2
        param_ascend, _ = torch.sort(param, dim=2, descending=False)
        param = torch.cat([param_ascend[:, :, :2], param_descend], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def get_weight2(self, param1, param2, param3):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2))
        param3 = (F.relu(param3) + F.relu(param3))
        param = torch.cat([param1, param2, param3], dim=2)
        param_descend, _ = torch.sort(param, dim=2, descending=True)
        param_descend[:, :, :2] = param_descend[:, :, :2] / 2
        param_ascend, _ = torch.sort(param, dim=2, descending=False)
        param = torch.cat([param_ascend[:, :, :2], param_descend], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def get_weight_2d(self, param1, param2, param3, param4, param5):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2))
        param3 = (F.relu(param3) + F.relu(param3))
        param4 = (F.relu(param4) + F.relu(param4))
        param5 = (F.relu(param5) + F.relu(param5))
        param = torch.cat([param1, param2, param3, param4, param5], dim=2)
        param_ascend, _ = torch.sort(param, dim=2, descending=False)
        ind = [0, 3, 4, 2, 1]
        param_sorted = param_ascend.scatter_(dim=-1, index=ind, src=param_ascend)

        return param

    def forward(self, x):
        input_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # weight = self.get_weight2(self.param1, self.param2, self.param3)
        weight = self.get_weight(self.param1, self.param2, self.param3)
        # weight = self.get_weight_2d(self.param1, self.param2, self.param3, self.param4, self.param5)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        output_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        x = x + input_mean.repeat(1, 1, x.size(2), x.size(3)) - output_mean.repeat(1, 1, x.size(2), x.size(3))
        return x

class ParamBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.20, mul=1)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.25, mul=3)
        self.param3 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.01, mul=5)

    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375, mul=1):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        param.data.normal_(mean=0, std=np.sqrt(2.0 / fan_out))   # 0.2, 0.05
        param *= mul
        # nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight(self, param1, param2, param3):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2)) / 3
        param3 = (F.relu(param3) + F.relu(param3)) / 5
        param = torch.cat([param3,
                           param3 + param2,
                           param3 + param2 + param1,
                           param3 + param2,
                           param3], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def get_weight_2d(self, param1, param2, param3):
        param = F.relu(param1) + F.relu(param1)
        return param

    def forward(self, x):
        weight = self.get_weight(self.param1, self.param2, self.param3)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x


class ParamBlurPool3x3(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0, mul=3)
        # self.param3 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.20)
        # self.param4 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.25, mul=3)

    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375, mul=1):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size], dtype=torch.float, requires_grad=True)
        # param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size],
        #                     dtype=torch.float, requires_grad = True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        param.data.normal_(mean=0, std=np.sqrt(2.0 / fan_out))   # 0.2, 0.05
        # param.data.uniform_(-np.sqrt(6.0 / fan_out), np.sqrt(6.0 / fan_out))
        param *= mul
        # nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_norm(self, weight, eps=1e-10):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / (weight_sum + eps)
        return normalized_weight


    def get_weight(self, param1, param2):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2)) / 3
        # param1 = torch.clamp(param1, min=0.0001)
        # param2 = torch.clamp(param2, min=0.0001)
        param = torch.cat([param2,
                           param2 + param1,
                           param2], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def get_weight2(self, param1, param2, param3, param4, eps=1e-5):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2)) / 3
        param3 = F.relu(param3) + F.relu(param3)
        param4 = (F.relu(param4) + F.relu(param4)) / 3
        param12 = torch.cat([param2,
                             param2 + param1,
                             param2], dim=2)
        param34 = torch.cat([param4,
                             param4 + param3,
                             param4], dim=2)
        param = torch.einsum('bci,bcj->bcij', param12, param34)

        return param

    def get_weight3(self, param1, param2):
        "relu activation"
        param1 = F.relu(param1)
        param2 = (F.relu(param2) / 3)
        # param1 = torch.clamp(param1, min=0.0001)
        # param2 = torch.clamp(param2, min=0.0001)
        param = torch.cat([param2,
                           param2 + param1,
                           param2], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def get_weight4(self, param1, param2):
        param1 = F.sigmoid(param1)
        param2 = F.sigmoid(param2 / 3)
        # param1 = torch.clamp(param1, min=0.0001)
        # param2 = torch.clamp(param2, min=0.0001)
        param = torch.cat([param2,
                           param2 + param1,
                           param2], dim=2)
        param = torch.einsum('bci,bcj->bcij', param, param)

        return param

    def forward(self, x):
        weight = self.get_weight4(self.param1, self.param2)
        x = self.reflection_pad(x)
        if self.groups == self.in_channels:
            weight = self.get_norm(weight)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x


class ParamBlurPool3x3_2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.20, mul=1)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.25, mul=5)
        self.param3 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.01, mul=9)

    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375, mul=1):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size],
                            dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        param.data.normal_(mean=0, std=np.sqrt(2.0 / fan_out))   # 0.2, 0.05
        param *= mul
        # nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight_2d(self, param1, param2, param3):
        param1 = F.relu(param1) + F.relu(param1)
        param2 = (F.relu(param2) + F.relu(param2)) / 5
        param3 = (F.relu(param3) + F.relu(param3)) / 9
        row1 = torch.cat([param3, param3 + param2, param3], dim=2)
        row2 = torch.cat([param3 + param2, param3 + param2 + param1, param3 + param2], dim=2)
        param = torch.cat([row1, row2, row1], dim=3)

        return param

    def forward(self, x):
        weight = self.get_weight_2d(self.param1, self.param2, self.param3)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x


class SigmaBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        # param.data.normal_(mean=0.4, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        nn.init.constant_(param, 0.375)
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        o = self.get_gaussian_inv(param, loc=0)
        x = self.get_gaussian_inv(param, loc=1)
        y = self.get_gaussian_inv(param, loc=2)
        z = self.get_gaussian_inv(param, loc=4)
        u = self.get_gaussian_inv(param, loc=5)
        v = self.get_gaussian_inv(param, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        return weight

    def get_gaussian_inv(self, b, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return b * b * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)

    def forward(self, x):
        weight = self.get_weight(self.param)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x

class NormalBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.140625)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.375)


    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        # param.data.normal_(mean=mean, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight(self, param1, param2):
        param1 = F.relu(param1) + F.relu(-param1)
        param2 = F.relu(param2) + F.relu(-param2)
        o = self.get_gaussian_inv(param1, param2, loc=0)
        x = self.get_gaussian_inv(param1, param2, loc=1)
        y = self.get_gaussian_inv(param1, param2, loc=2)
        z = self.get_gaussian_inv(param1, param2, loc=4)
        u = self.get_gaussian_inv(param1, param2, loc=5)
        v = self.get_gaussian_inv(param1, param2, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        return weight

    def get_gaussian_inv(self, a, b, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return a * a * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)
        # return a * torch.exp(-loc * math.pi * b * b)

    def get_gaussian_inv2(self, a, b, c, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return a * a * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)
        # return a * torch.exp(-loc * math.pi * b * b)

    def forward(self, x):
        weight = self.get_weight(self.param1, self.param2)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x


class NormalBlurPool2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.3989)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.3989)
        self.param3 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.7071)
        self.param4 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.7071)


    def get_param(self, in_channels, out_channels, kernel_size, groups, mean=0.375):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        # param.data.normal_(mean=mean, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight(self, a, b, c, d):
        param1 = F.relu(a * b) + F.relu(-a * b)
        param2 = F.relu(c * d) + F.relu(-c * d)
        o = self.get_gaussian_inv2(param1, param2, loc=0)
        x = self.get_gaussian_inv2(param1, param2, loc=1)
        y = self.get_gaussian_inv2(param1, param2, loc=2)
        z = self.get_gaussian_inv2(param1, param2, loc=4)
        u = self.get_gaussian_inv2(param1, param2, loc=5)
        v = self.get_gaussian_inv2(param1, param2, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        return weight

    def get_gaussian_inv(self, a, b, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return a * a * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)
        # return a * torch.exp(-loc * math.pi * b * b)

    def get_gaussian_inv2(self, a, b, loc):
        return a * torch.exp(-loc * b)

    def forward(self, x):
        weight = self.get_weight(self.param1, self.param2, self.param3, self.param4)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x


class NormalSumBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.140625)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups, 0.375)

    def get_param(self, in_channels, out_channels, kernel_size, groups, mean):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        # param.data.normal_(mean=0.375, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        nn.init.constant_(param, mean)
        return nn.Parameter(param)

    def get_weight(self, param1, param2):
        param1 = F.relu(param1) + F.relu(-param1)
        param2 = F.relu(param2) + F.relu(-param2)
        o = self.get_gaussian_inv(param1, param2, loc=0)
        x = self.get_gaussian_inv(param1, param2, loc=1)
        y = self.get_gaussian_inv(param1, param2, loc=2)
        z = self.get_gaussian_inv(param1, param2, loc=4)
        u = self.get_gaussian_inv(param1, param2, loc=5)
        v = self.get_gaussian_inv(param1, param2, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        return weight

    def get_gaussian_inv(self, a, b, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return a * a * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)

    def normalize_by_sum(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def normalize_by_center(self, weight):
        center = weight[:, :, 2:3, 2:3]
        center = center.repeat((1, 1, 5, 5))
        normalized_weight = weight / center
        return normalized_weight

    def forward(self, x):
        weight = self.get_weight(self.param1, self.param2)
        weight_norm = self.normalize_by_sum(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x


class NormalCenterBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # std = np.sqrt(0.05 / fan_out)
        # param.data.normal_(mean=0.375, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        nn.init.constant_(param, 0.375)
        return nn.Parameter(param)

    def get_weight(self, param1, param2):
        param1 = F.relu(param1) + F.relu(-param1)
        param2 = F.relu(param2) + F.relu(-param2)
        o = self.get_gaussian_inv(param1, param2, loc=0)
        x = self.get_gaussian_inv(param1, param2, loc=1)
        y = self.get_gaussian_inv(param1, param2, loc=2)
        z = self.get_gaussian_inv(param1, param2, loc=4)
        u = self.get_gaussian_inv(param1, param2, loc=5)
        v = self.get_gaussian_inv(param1, param2, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        return weight

    def get_gaussian_inv(self, a, b, loc):
        # return b * b * torch.exp(-loc * math.pi * b * b)
        return a * a * torch.exp(-loc * math.pi * b * b)
        # return b * torch.exp(-loc * math.pi * b * b)

    def normalize_by_sum(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def normalize_by_center(self, weight):
        center = weight[:, :, 2:3, 2:3]
        center = center.repeat((1, 1, 5, 5))
        normalized_weight = weight / center
        return normalized_weight

    def forward(self, x):
        weight = self.get_weight(self.param1, self.param2)
        weight_norm = self.normalize_by_center(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x



class SigmaNormBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.5, std=np.sqrt(2.0 / fan_out))
        # param.data.normal_(mean=0.0, std=np.sqrt(6.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        # x = self.get_gaussian(param, loc=0)
        # y = self.get_gaussian(param, loc=1)
        # z = self.get_gaussian(param, loc=2)
        x = self.get_gaussian_inv(param, loc=0)
        y = self.get_gaussian_inv(param, loc=1)
        z = self.get_gaussian_inv(param, loc=2)
        row1 = torch.cat([z, y, z], dim=2)
        row2 = torch.cat([y, x, y], dim=2)
        weight = torch.cat([row1, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def get_gaussian_inv(self, b, loc):
        return b * torch.exp(-loc * math.pi * b * b)

    def normalize_weight(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def forward(self, x):
        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x


class SigmaNormBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.5, std=np.sqrt(2.0 / fan_out))
        # param.data.normal_(mean=0.0, std=np.sqrt(6.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        # x = self.get_gaussian(param, loc=0)
        # y = self.get_gaussian(param, loc=1)
        # z = self.get_gaussian(param, loc=2)
        x = self.get_gaussian_inv(param, loc=0)
        y = self.get_gaussian_inv(param, loc=1)
        z = self.get_gaussian_inv(param, loc=2)
        row1 = torch.cat([z, y, z], dim=2)
        row2 = torch.cat([y, x, y], dim=2)
        weight = torch.cat([row1, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def get_gaussian_inv(self, b, loc):
        return b * torch.exp(-loc * math.pi * b * b)

    def normalize_weight(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def forward(self, x):
        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x


class SigmaNormBlurPool5x5(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.49, std=np.sqrt(0.05 / fan_out))
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        o = self.get_gaussian_inv(param, loc=0)
        x = self.get_gaussian_inv(param, loc=1)
        y = self.get_gaussian_inv(param, loc=2)
        z = self.get_gaussian_inv(param, loc=4)
        u = self.get_gaussian_inv(param, loc=5)
        v = self.get_gaussian_inv(param, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def get_gaussian_inv(self, b, loc):
        return 1 / math.pi * b * torch.exp(-loc * b)

    def normalize_weight(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def forward(self, x):
        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x


class SigmaCenterNormBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.5, std=np.sqrt(2.0 / fan_out))
        # param.data.normal_(mean=0.0, std=np.sqrt(6.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        x = self.get_gaussian_inv(param, loc=0)
        y = self.get_gaussian_inv(param, loc=1)
        z = self.get_gaussian_inv(param, loc=2)
        row1 = torch.cat([z, y, z], dim=2)
        row2 = torch.cat([y, x, y], dim=2)
        weight = torch.cat([row1, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def get_gaussian_inv(self, b, loc):
        return b * torch.exp(-loc * math.pi * b * b)

    def normalize_weight(self, weight):
        center = weight[:, :, 1:2, 1:2]
        center = center.repeat((1, 1, 3, 3))
        normalized_weight = weight / center
        return normalized_weight


    def forward(self, x):

        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)

        # weight = self.get_weight5x5(self.param)
        # weight_norm = self.normalize_weight5x5(weight)
        # x = self.reflection_pad2(x)
        # x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x


class SigmaCenterNormBlurPool5x5(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.reflection_pad = nn.ReflectionPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.49, std=np.sqrt(0.05 / fan_out))   # 0.2, 0.05
        return nn.Parameter(param)

    def get_weight(self, param):
        param = F.relu(param) + F.relu(-param)
        o = self.get_gaussian_inv(param, loc=0)
        x = self.get_gaussian_inv(param, loc=1)
        y = self.get_gaussian_inv(param, loc=2)
        z = self.get_gaussian_inv(param, loc=4)
        u = self.get_gaussian_inv(param, loc=5)
        v = self.get_gaussian_inv(param, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def get_gaussian_inv(self, b, loc):
        # return b * torch.exp(-loc * math.pi * b * b)
        return b * torch.exp(-loc * math.pi * b)

    def normalize_weight(self, weight):
        center = weight[:, :, 2:3, 2:3]
        center = center.repeat((1, 1, 5, 5))
        normalized_weight = weight / center
        return normalized_weight

    def forward(self, x):

        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight_norm, stride=self.stride, groups=self.groups)
        return x