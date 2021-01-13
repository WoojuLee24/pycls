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
        if kernel_size == 3:
            self.kernel = self.get_weight3x3(self.param)
        else:
            self.kernel = self.get_weight5x5(self.param)

    def get_param(self, in_channels, out_channels, kernel_size, groups, sigma=0.65):
        param = torch.ones([out_channels, in_channels // groups, 1, 1], dtype=torch.float,
                            requires_grad=False)
        param = param.cuda()
        param = sigma * param

        return nn.Parameter(param)

    def get_weight3x3(self, param):
        x = self.get_gaussian(param, loc=0)
        y = self.get_gaussian(param, loc=1)
        z = self.get_gaussian(param, loc=2)
        row1 = torch.cat([z, y, z], dim=2)
        row2 = torch.cat([y, x, y], dim=2)
        weight = torch.cat([row1, row2, row1], dim=3)
        weight = self.normalize_sum_weight3x3(weight)

        return weight

    def get_weight5x5(self, param):
        o = self.get_gaussian(param, loc=0)
        x = self.get_gaussian(param, loc=1)
        y = self.get_gaussian(param, loc=2)
        z = self.get_gaussian(param, loc=4)
        u = self.get_gaussian(param, loc=5)
        v = self.get_gaussian(param, loc=8)
        row1 = torch.cat([v, u, z, u, v], dim=2)
        row2 = torch.cat([u, y, x, y, u], dim=2)
        row3 = torch.cat([z, x, o, x, z], dim=2)
        weight = torch.cat([row1, row2, row3, row2, row1], dim=3)
        weight = self.normalize_sum_weight5x5(weight)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def normalize_center_weight5x5(self, weight):
        center = weight[:, :, 2:3, 2:3]
        center = center.repeat((1, 1, 5, 5))
        normalized_weight = weight / center

        return normalized_weight

    def normalize_sum_weight5x5(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def normalize_sum_weight3x3(self, weight):
        weight_sum = weight.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        normalized_weight = weight / weight_sum
        return normalized_weight

    def forward(self, x):
        x = self.reflection_pad(x)
        x = F.conv2d(x, self.kernel, stride=self.stride, groups=self.groups)
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
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float,
                            requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.8, std=np.sqrt(2.0 / fan_out))
        # param.data.normal_(mean=0.0, std=np.sqrt(6.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight(self, param):
        x = self.get_gaussian(param, loc=0)
        y = self.get_gaussian(param, loc=1)
        z = self.get_gaussian(param, loc=2)
        row1 = torch.cat([z, y, z], dim=2)
        row2 = torch.cat([y, x, y], dim=2)
        weight = torch.cat([row1, row2, row1], dim=3)

        return weight

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)


    def forward(self, x):
        weight = self.get_weight(self.param)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x

class AbsSigmaBlurPool(nn.Conv2d):
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

    def forward(self, x):
        weight = self.get_weight(self.param)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
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
        param.data.normal_(mean=0.2, std=np.sqrt(0.05 / fan_out))
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
        return b * torch.exp(-loc * math.pi * b * b)

    def normalize_weight(self, weight):
        center = weight[:, :, 2:3, 2:3]
        center = center.repeat((1, 1, 5, 5))
        normalized_weight = weight / center
        return normalized_weight

    def forward(self, x):

        weight = self.get_weight(self.param)
        weight_norm = self.normalize_weight(weight)
        x = self.reflection_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x