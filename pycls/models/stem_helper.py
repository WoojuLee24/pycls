from pycls.core.config import cfg
from pycls.models.blocks import (
    SE,
    activation,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
)
import torch
from torch.nn import Module
from pycls.models.endstop_helper import *


class ResStemCifar(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDivide3x3ConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDivide3x3ConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDivide3x3(w_out, w_out, 3, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDivide5x5ConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDivide5x5ConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDivide5x5(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarSlope5x5ConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarSlope5x5ConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingSlope(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDoG5x5ConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDoG5x5ConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDoG5x5(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDivide5x5GroupEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDivide5x5GroupEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDivide5x5(w_out, w_out*4, 5, stride=1, groups=w_out)
        self.conv2 = conv2d(w_out*4, w_out, 1, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.conv2(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDivide5x5ConvOnlyDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDivide5x5ConvOnlyDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDivide5x5(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarDilationConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarDilationConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = EndstoppingDilation(w_out, w_out, 3, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx



class ResStemCifarCompareFixedSmConvDcEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompareFixedSmConvDcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = CompareFixedSM(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarCompareFixedSmConvDcEntireNoaf(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompareFixedSmConvDcEntireNoaf, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.e = CompareFixedSM(w_out, w_out, 5, stride=1, groups=w_out)
        self.af = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifar7x7ConvFcBnEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar7x7ConvFcBnEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarCompare3x3x2ConvFcBnEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x2ConvFcBnEntire, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3)
        self.bn2 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)

        return cx


class ResStemCifarCompare3x3x3ConvFcBnEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x3ConvFcBnEntire, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3)
        self.bn2 = norm2d(w_out)
        self.conv3 = conv2d(w_out, w_out, 3)
        self.bn3 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarCompare3x3x2ConvDcBnEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x2ConvDcBnEntire, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, groups=w_out)
        self.bn2 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)

        return cx


class ResStemCifarCompare3x3x3ConvDcBnEntire(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x3ConvDcBnEntire, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, groups=w_out)
        self.bn2 = norm2d(w_out)
        self.conv3 = conv2d(w_out, w_out, 3 ,groups=w_out)
        self.bn3 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifarCompare3x3x2ConvDcBnEntire1Act(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x2ConvDcBnEntire1Act, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, groups=w_out)
        self.bn2 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)

        return cx


class ResStemCifarCompare3x3x3ConvDcBnEntire1Act(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifarCompare3x3x3ConvDcBnEntire1Act, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 3)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, groups=w_out)
        self.bn2 = norm2d(w_out)
        self.conv3 = conv2d(w_out, w_out, 3, groups=w_out)
        self.bn3 = norm2d(w_out)
        self.af = activation()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3, groups=w_out)
        cx = norm2d_cx(cx, w_out)

        return cx


class ResStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemCompare3x3SeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemCompare3x3SeparationEntire, self).__init__()
        self.conv1 = conv2d(w_in, w_out, 7, stride=2)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, stride=1, groups=w_out)
        self.bn2 = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemCompareDilationSeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemCompareDilationSeparationEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.compare = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, groups=w_out, bias=False, padding_mode='replicate')
        self.compare_dilation = nn.Conv2d(w_out, w_out, 3, stride=1, padding=2, dilation=2, groups=w_out, bias=False, padding_mode='replicate')
        self.pool = pool2d(w_out, 3, stride=2)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x1 = self.compare(x)
        x2 = self.compare_dilation(x)
        x = x1+x2
        x = self.af(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx

class ResStemCompareDilationSeparationBnEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemCompareDilationSeparationBnEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.compare = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, groups=w_out, bias=False)
        self.compare_dilation = nn.Conv2d(w_out, w_out, 3, stride=1, padding=2, dilation=2, groups=w_out, bias=False)
        self.e_bn = norm2d(w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x1 = self.compare(x)
        x2 = self.compare_dilation(x)
        x = x1+x2
        x = self.e_bn(x)
        x = self.af(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemCompare3x3x3ConvFcBnEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemCompare3x3x3ConvFcBnEntire, self).__init__()
        # self.conv = conv2d(w_in, w_out, 7, stride=2)
        # self.bn = norm2d(w_out)
        self.conv1 = conv2d(w_in, w_out, 3, stride=1)
        self.bn1 = norm2d(w_out)
        self.conv2 = conv2d(w_out, w_out, 3, stride=1)
        self.bn2 = norm2d(w_out)
        self.conv3 = conv2d(w_out, w_out, 3, stride=2)
        self.bn3 = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=1)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=1)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilation(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilation, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)
        self.e = EndstoppingDilation(w_out, w_out, 3, stride=1, groups=1)
        self.e_bn = norm2d(w_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.pool(x)
        xe = self.e(x)
        xe = self.e_bn(xe)
        xe = self.af(xe)
        x = torch.cat((x, xe), dim=1)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationFcEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationFcEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilation(w_out, w_out, 3, stride=1, groups=1)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationSeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationSeparationEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilation(w_out, w_out, 3, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationGroupEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationGroupEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilation(w_out, w_out* 4, 3, stride=1, groups=w_out)
        self.conv2 = conv2d(w_out * 4, w_out, 1, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationSeparationBnEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationSeparationBnEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilation(w_out, w_out, 3, stride=1, groups=w_out)
        self.e_bn = norm2d(w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.e_bn(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationPreluSeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationPreluSeparationEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilationPReLU(w_out, w_out, 3, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDilationPreluSeparationBnEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDilationPreluSeparationBnEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDilationPReLU(w_out, w_out, 3, stride=1, groups=w_out)
        self.e_bn = norm2d(w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.e_bn(x)
        x = self.af(x)
        x = self.pool(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemFixSMEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemFixSMEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = CompareFixedSM(w_out, w_out, 5, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDivideSeparation(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivideSeparation, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)
        self.e = EndstoppingDivide(w_out, w_out, 3, stride=1, groups=w_out)
        self.e_bn = norm2d(w_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        xe = self.e(x)
        xe = self.af(xe)
        x = torch.cat((x, xe), dim=1)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDivide3x3SeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivide3x3SeparationEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDivide3x3(w_out, w_out, 3, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDivide5x5SeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivide5x5SeparationEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDivide5x5(w_out, w_out, 5, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDivide3x3GroupEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivide3x3GroupEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDivide3x3(w_out, w_out*4, 3, stride=1, groups=w_out)
        self.conv2 = conv2d(w_out*4, w_out, 1, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResStemEndstopDivide5x5GroupEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivide5x5GroupEntire, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.e = EndstoppingDivide5x5(w_out, w_out*4, 5, stride=1, groups=w_out)
        self.conv2 = conv2d(w_out*4, w_out, 1, stride=1, groups=w_out)
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.e(x)
        x = self.af(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class EndstopDilationStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(EndstopDilationStem, self).__init__()
        self.e = EndstoppingDilation(w_in, w_out, 3, stride=1, groups=1)
        self.e_bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        x = self.e(x)
        x = self.e_bn(x)
        x = self.af(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class SimpleStem(Module):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx