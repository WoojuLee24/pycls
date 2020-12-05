import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class EndstoppingDivide3x3(nn.Conv2d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
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

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

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


class EndstoppingDivide5x5(nn.Conv2d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.replication_pad = nn.ReplicationPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.sm = self.get_sm(self.in_channels, self.out_channels, self.groups, mul=1e-3)
        # self.center_threshold, self.surround_threshold = self.get_threshold_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_sm(self, in_channels, out_channels, groups, mul=1e-3):
        sm = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.18, 0.49, 1, 0.49, -0.18],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        sm = sm * mul
        sm = sm.repeat((out_channels, in_channels // groups, 1, 1))
        return sm

    def get_threshold_param(self, in_channels, out_channels, kernel_size, groups):

        center = torch.tensor([[0, 0, 0, 0, 0],
                               [0, 0.017, 0.049, 0.017, 0],
                               [0, 0.049, 0.1, 0.049, 0],
                               [0, 0.017, 0.049, 0.017, 0],
                               [0, 0, 0, 0, 0]], requires_grad=False).cuda()
        center = center.repeat(out_channels, in_channels//groups, 1, 1)

        surround = torch.tensor([[-0.027, -0.023, -0.018, -0.023, -0.027],
                               [-0.023, 0, 0, 0, -0.023],
                               [-0.018, 0, 0, 0, -0.018],
                               [-0.023, 0, 0, 0, -0.023],
                               [-0.027, -0.023, -0.018, -0.023, -0.027]], requires_grad=False).cuda()
        surround = surround.repeat(out_channels, in_channels // groups, 1, 1)

        return center, surround

    def get_weight_5x5(self, param):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, 1:4, 1:4], (1, 1, 1, 1))
        surround = param - center
        surround = surround * 9/16
        center = F.relu(center) + F.relu(-center)
        surround = - F.relu(surround) - F.relu(-surround)
        # center = torch.max(center, self.center_threshold)
        # surround = torch.min(surround, self.surround_threshold)

        weight = center + surround
        weight = weight + self.sm

        return weight


    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def forward(self, x):
        weight = self.get_weight_5x5(self.param)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        # x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return x


class SurroundDivide(nn.Conv2d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.replication_pad = nn.ReplicationPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        # self.sm = self.get_sm(self.in_channels, self.out_channels, self.groups, mul=1e-3)
        # self.center_threshold, self.surround_threshold = self.get_threshold_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_sm(self, in_channels, out_channels, groups, mul=1e-3):
        sm = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.18, 0.49, 1, 0.49, -0.18],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        sm = sm * mul
        sm = sm.repeat((out_channels, in_channels // groups, 1, 1))
        return sm


    def get_weight_5x5(self, param):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.relu(param) + F.relu(-param)
        center_sum = center.sum(dim=3).sum(dim=2)
        center_mean = center_sum / 16
        sur = -center_mean
        sur2 = torch.stack([sur, sur, sur], dim=2)
        sur2 = torch.stack([sur2, sur2, sur2], dim=3)
        sur2 = F.pad(sur2, (1, 1, 1, 1))
        sur = torch.stack([sur, sur, sur, sur, sur], dim=2)
        surround = torch.stack([sur, sur, sur, sur, sur], dim=3)
        surround = surround - sur2
        center = F.pad(center, (1, 1, 1, 1))
        weight = center + surround

        return weight


    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def forward(self, x):
        weight = self.get_weight_5x5(self.param)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        # x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return x


class EndstoppingDoG5x5(nn.Conv2d):

    """
    End-stopping Difference of Gaussian kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.replication_pad = nn.ReplicationPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, 2], dtype=torch.float, requires_grad=True)
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
        a, b = param[:, :, 0], param[:, :, 1]
        # a - b
        x = a * torch.exp(-a * a) - b * torch.exp(-b * b)
        y = a * torch.exp(-2 * a * a) - b * torch.exp(-2 * b * b)
        z = a * torch.exp(-4 * a * a) - b * torch.exp(-4 * b * b)
        u = a * torch.exp(-5 * a * a) - b * torch.exp(-5 * b * b)
        v = a * torch.exp(-8 * a * a) - b * torch.exp(-8 * b * b)
        # weight = torch.cat([v, u, z, u, v], dim=)
        weight = torch.tensor([[v, u, z, u, v], [u, y, x, y, u], [z, x, a-b, x, z], [u, y, x, y, u], [v, u, z, u, v]])

        return weight


    def forward(self, x):
        weight = self.get_weight_5x5(self.param)
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
        self.param1 = self.get_param( self.in_channels, self.out_channels, self.kernel_size, self.groups)
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


class ComparingDilation(nn.Conv2d):

    """
    Comparing dilation kernel for solving aperture problem
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = kernel_size * kernel_size * out_channels
        param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def forward(self, x):
        x1 = F.conv2d(x, self.param1, stride=self.stride, padding=1, dilation=1, groups=1)
        x2 = F.conv2d(x, self.param2, stride=self.stride, padding=2, dilation=2, groups=1)
        x = x1 + x2

        return x


class EndstoppingSlope(nn.Conv2d):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups, mean=-1.35)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups, mean=-1.35)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.replication_pad = nn.ReplicationPad2d(2)

    def get_param(self, in_channels, out_channels, groups, mean=0.0):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('sigmoid'))
        param = param - mean
        return nn.Parameter(param)

    def get_weight(self, slope_x, slope_y, center):
        one = torch.ones([self.out_channels, self.in_channels // self.groups, 1, 1], dtype=torch.float).cuda()
        bias = 1 / 2 * (torch.sigmoid(center) + one / 2)
        # bias = one
        kernel_x = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one / 2),
                              bias - 1 / 2 * (torch.sigmoid(slope_x) + one / 2),
                              bias,
                              bias - 1 / 2 * (torch.sigmoid(slope_x) + one / 2),
                              bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one / 2)], dim=2)
        kernel_x = kernel_x.repeat((1, 1, 1, 5))
        kernel_y = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one / 2),
                              bias - 1 / 2 * (torch.sigmoid(slope_y) + one / 2),
                              bias,
                              bias - 1 / 2 * (torch.sigmoid(slope_y) + one / 2),
                              bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one / 2)], dim=3)
        kernel_y = kernel_y.repeat((1, 1, 5, 1))
        kernel = kernel_x + kernel_y
        return kernel

    def forward(self, x):
        weight = self.get_weight(self.slope_x, self.slope_y, self.center)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=1, groups=self.groups)
        return x


class EndstoppingSlopeTanh(nn.Conv2d):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups, mean=-0.675)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups, mean=-0.675)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.replication_pad = nn.ReplicationPad2d(2)

    def get_param(self, in_channels, out_channels, groups, mean=0):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('tanh'))
        param = param - mean
        return nn.Parameter(param)

    def get_weight(self, slope_x, slope_y, center):
        one = torch.ones([self.out_channels, self.in_channels // self.groups, 1, 1], dtype=torch.float).cuda()
        bias = 1 / 4 * (torch.tanh(center/2) + 2 * one)
        # bias = one
        kernel_x = torch.cat([bias - 2 * 1 / 4 * (torch.tanh(slope_x/2) + 2 * one),
                              bias - 1 / 4 * (torch.tanh(slope_x/2) + 3 * one),
                              bias,
                              bias - 1 / 4 * (torch.tanh(slope_x/2) + 2 * one),
                              bias - 2 * 1 / 4 * (torch.tanh(slope_x/2) + 2 * one)], dim=2)
        kernel_x = kernel_x.repeat((1, 1, 1, 5))
        kernel_y = torch.cat([bias - 2 * 1 / 4 * (torch.tanh(slope_y/2) + 2 * one),
                              bias - 1 / 4 * (torch.tanh(slope_y/2) + 2 * one),
                              bias,
                              bias - 1 / 4 * (torch.tanh(slope_y/2) + 2 * one),
                              bias - 2 * 1 / 4 * (torch.tanh(slope_y/2) + 2 * one)], dim=3)
        kernel_y = kernel_y.repeat((1, 1, 5, 1))
        kernel = kernel_x + kernel_y
        return kernel

    def forward(self, x):
        weight = self.get_weight(self.slope_x, self.slope_y, self.center)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=1, groups=self.groups)
        return x


class EndstoppingSlopeRelu(nn.Conv2d):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups, mean=0.35)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups, mean=0.35)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups, mean=1)
        self.replication_pad = nn.ReplicationPad2d(2)

    def get_param(self, in_channels, out_channels, groups, mean=0.0):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        fan_out = 1 * 1 * out_channels
        param.data.normal_(mean=mean, std=np.sqrt(2.0 / fan_out))
        return nn.Parameter(param)

    def get_constraint(self, param, min=0.5, max=1):
        return torch.clamp(F.relu(param), min=min, max=max)

    def get_weight(self, slope_x, slope_y, center):
        slope_x = self.get_constraint(slope_x)
        slope_y = self.get_constraint(slope_y)
        bias = self.get_constraint(center)

        kernel_x = torch.cat([bias - 2 * slope_x,
                              bias - 1 * slope_x,
                              bias,
                              bias - 1 * slope_x,
                              bias - 2 * slope_x], dim=2)
        kernel_x = kernel_x.repeat((1, 1, 1, 5))
        kernel_y = torch.cat([- 2 * slope_y,
                              - 1 * slope_y,
                              0,
                              - 1 * slope_y,
                              - 2 * slope_y], dim=3)
        kernel_y = kernel_y.repeat((1, 1, 5, 1))
        kernel = kernel_x + kernel_y
        return kernel

    def forward(self, x):
        weight = self.get_weight(self.slope_x, self.slope_y, self.center)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=1, groups=self.groups)
        return x


class EndstoppingSlopeTanh2(nn.Conv2d):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.replication_pad = nn.ReplicationPad2d(2)

    def get_param(self, in_channels, out_channels, groups):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('tanh'))
        return nn.Parameter(param)

    def get_weight(self, slope_x, slope_y, center):
        one = torch.ones([self.out_channels, self.in_channels // self.groups, 1, 1], dtype=torch.float).cuda()
        bias = 1 / 4 * (torch.tanh(center) + 3 * one)
        kernel_x = torch.cat([bias - pow(2, 3/2) * 1 / 4 * (torch.tanh(slope_x) + 3 * one),
                              bias - 1 / 4 * (torch.tanh(slope_x) + 3 * one),
                              bias,
                              bias - 1 / 4 * (torch.tanh(slope_x) + 3 * one),
                              bias - pow(2, 3/2) * 1 / 4 * (torch.tanh(slope_x) + 3 * one)], dim=2)
        kernel_x = kernel_x.repeat((1, 1, 1, 5))
        kernel_y = torch.cat([bias - pow(2, 3/2) * 1 / 4 * (torch.tanh(slope_y) + 3 * one),
                              bias - 1 / 4 * (torch.tanh(slope_y) + 3 * one),
                              bias,
                              bias - 1 / 4 * (torch.tanh(slope_y) + 3 * one),
                              bias - pow(2, 3/2) * 1 / 4 * (torch.tanh(slope_y) + 3 * one)], dim=3)
        kernel_y = kernel_y.repeat((1, 1, 5, 1))
        kernel = kernel_x + kernel_y
        return kernel

    def forward(self, x):
        weight = self.get_weight(self.slope_x, self.slope_y, self.center)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=1, groups=self.groups)
        return x



class EndstoppingSigmoid(nn.Conv2d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.replication_pad = nn.ReplicationPad2d(2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.sm = self.get_sm(self.in_channels, self.out_channels, groups, range=0.25)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size, kernel_size], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('sigmoid'))
        return nn.Parameter(param)

    def get_sm(self, in_channels, out_channels, groups, range):
        sm = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.18, 0.49, 1, 0.49, -0.18],
                      [-0.23, 0.17, 0.49, 0.17, -0.23],
                      [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        sm = sm - range
        sm = sm.repeat((out_channels, in_channels // groups, 1, 1))
        return sm

    def get_weight_5x5(self, param):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        weight = self.sm + 1 / 2 * torch.sigmoid(param)
        return weight


    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def forward(self, x):
        weight = self.get_weight_5x5(self.param)
        x = self.replication_pad(x)
        x = F.conv2d(x, weight, stride=self.stride, groups=self.groups)
        return x