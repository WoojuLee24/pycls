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
        self.replication_pad = nn.ReplicationPad2d(1)
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
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        # x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
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
        self.replication_pad1 = nn.ReplicationPad2d(1)
        self.replication_pad2 = nn.ReplicationPad2d(2)

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
        x1 = self.replication_pad1(x)
        x2 = self.replication_pad2(x)
        x1 = F.conv2d(x1, center, stride=self.stride, dilation=1, groups=self.groups)
        x2 = F.conv2d(x2, surround, stride=self.stride, dilation=2, groups=self.groups)
        x = x1 + x2
        return x


class EndstoppingDilationPReLU(nn.Conv2d):

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
        self.prelu_center = nn.PReLU()
        self.prelu_surround = nn.PReLU()
        self.replication_pad1 = nn.ReplicationPad2d(1)
        self.replication_pad2 = nn.ReplicationPad2d(2)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_center(self, param):
        center = self.prelu_center(param)
        return center

    def get_surround(self, param):
        center = F.pad(param[:, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround = - self.prelu_surround(-surround)
        return surround

    def forward(self, x):
        center = self.get_center(self.param1)
        surround = self.get_surround(self.param2)
        x1 = self.replication_pad1(x)
        x2 = self.replication_pad2(x)
        x1 = F.conv2d(x1, center, stride=self.stride, dilation=1, groups=self.groups)
        x2 = F.conv2d(x2, surround, stride=self.stride, dilation=2, groups=self.groups)
        x = x1 + x2
        return x


class EndstoppingDilationPReLUOld(nn.Conv2d):

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
        self.prelu_center = nn.PReLU()
        self.prelu_surround = nn.PReLU()

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_center(self, param):
        center = self.prelu_center(param)
        return center

    def get_surround(self, param):
        center = F.pad(param[:, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround = - self.prelu_surround(-surround)
        return surround

    def forward(self, x):
        center = self.get_center(self.param1)
        surround = self.get_surround(self.param2)
        x1 = F.conv2d(x, center, stride=self.stride, dilation=1, padding=self.padding, groups=self.groups)
        x2 = F.conv2d(x, surround, stride=self.stride, dilation=2, padding=(2, 2), groups=self.groups)
        x = x1 + x2
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
        kernel = kernel.repeat(out_channels, in_channels//groups, 1, 1)

        return kernel

    def forward(self, x):
        x = F.conv2d(x, self.param, stride=self.stride, padding=self.padding, groups=self.groups)
        return x

