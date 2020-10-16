import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class EndstoppingDivide(nn.Conv2d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight_5x5(self, param):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, 1:4, 1:4], (1, 1, 1, 1))
        surround = param - center
        surround = surround * 9/16
        weight = F.relu(center) + F.relu(-center) - F.relu(surround) - F.relu(-surround)
        return weight

    def get_weight_3x3(self, param):
        """
        3x3 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround /= 8
        weight = F.relu(center) + F.relu(-center) - F.relu(surround) - F.relu(-surround)
        return weight

    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def forward(self, x):
        weight = self.get_weight_3x3(self.param)
        x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return x


class EndstoppingDilation(nn.Conv2d):

    """
    End-stopping dilation kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def get_surround(self, param):
        center = F.pad(param[:, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround = -F.relu(surround) - F.relu(-surround)
        return surround

    def forward(self, x):
        center = self.get_center(self.param1)
        surround = self.get_surround(self.param2)
        x1 = F.conv2d(x, center, stride=self.stride, dilation=1, padding=self.padding, groups=self.groups)
        x2 = F.conv2d(x, surround, stride=self.stride, dilation=2, padding=(2, 2), groups=self.groups)
        x = x1 + x2
        return x


class CompareDoG(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.conv3d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)

    def get_name(self):
        return type(self).__name__

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

