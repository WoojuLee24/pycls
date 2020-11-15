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


class ResStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=3)
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


class ResStemCompare(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemCompare, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)
        self.compare = conv2d(w_out, w_out, 3, stride=1, groups=w_out)
        self.compare_bn = norm2d(w_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        xe = self.compare(x)
        xe = self.compare_bn(xe)
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

class ResStemEndstopDivideSeparationEntire(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemEndstopDivideSeparationEntire, self).__init__()
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