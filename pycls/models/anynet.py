#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

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


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
        "res_stem_endstop_dilation": ResStemEndstopDilation,
        "endstop_dilation_stem": EndstopDilationStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
        "basic_res_bottleneck_block": BasicResBottleneckBlock,
        "endstop_dilation_res_bottleneck_block": EndstopDilationResBottleneckBlock,
        "endstop_divide_res_bottleneck_block": EndstopDivideResBottleneckBlock,
        "endstop_dilation_prelu_res_bottleneck_block": EndstopDilationPReLUResBottleneckBlock,
        "endstop_dilation_with_conv_res_bottleneck_block": EndstopDilationWithConvResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(AnyHead, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, num_classes):
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx

class BasicAnyHead(Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(BasicAnyHead, self).__init__()
        self.num_pathways = len(w_in)
        for pathway in range(self.num_pathways):
            self.avg_pool = gap2d(w_in)
            self.add_module("pathway{}_avgpool".format(pathway), self.avg_pool)
        self.fc = linear(sum(w_in), num_classes, bias=True)

    def forward(self, x):
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(x[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, H, W) -> (N, H, W, C).
        x = x.permute((0, 2, 3, 1))
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def complexity(cx, w_in, num_classes):
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, sum(w_in), num_classes, bias=True)
        return cx


class VanillaBlock(Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class BasicTransform(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBasicBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class EndstopDilationBasicTransform(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(EndstopDilationBasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = EndstoppingDilation(w_out, w_out, 3, stride=stride, groups=w_out)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class EndstopDilationResBasicBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstoppingDilationBasicTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = EndstoppingDilationBasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }


class BasicBottleneckTransform(Module):
    """

    Bottleneck transformation: 1x1, 3x3, 1x1.
    """

    def __init__(self, w_in, w_out, stride):
        super(BasicBottleneckTransform, self).__init__()
        w_b = int(round(w_out * 0.25))
        groups = w_b
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_af(x)
        x = self.c(x)
        x = self.c_bn(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride):
        w_b = int(round(w_out * 0.25))
        groups = w_b
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class BasicResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride):
        super(BasicResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicBottleneckTransform(w_in, w_out, stride)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BasicBottleneckTransform.complexity(cx, w_in, w_out, stride)
        return cx


    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }


class EndstopDilationBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        # self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b = EndstoppingDilation(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EndstopDilationResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstopDilationBottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = EndstopDilationBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class EndstopDilationPReLUBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationPReLUBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        # self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b = EndstoppingDilationPReLU(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EndstopDilationPReLUResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationPReLUResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstopDilationPReLUBottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = EndstopDilationPReLUBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class EndstopDilationBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        # self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b = EndstoppingDilation(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EndstopDilationWithConvResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationWithConvResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstopDilationWithConvBottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = EndstopDilationWithConvBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class EndstopDilationWithConvBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationWithConvBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()

        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)

        self.be = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.be_bn = norm2d(w_b)
        self.be_af = activation()

        self.e = EndstoppingDilation(w_b, w_b, 3, stride=1, groups=w_b)
        self.e_bn = norm2d(w_b)

        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)

        xe = self.be(x)
        xe = self.be_bn(xe)
        xe = self.be_af(xe)
        xe = self.e(xe)
        xe = self.e_bn(xe)

        x = self.b(x)
        x = self.b_bn(x)
        x = self.be_af(x+xe)

        x = self.c(x)
        x = self.c_bn(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx



class EndstopDivideBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDivideBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        # self.b = EndstoppingDivide(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        # self.c = conv2d(w_b, w_out, 1)
        self.c = EndstoppingDivide(w_b, w_out, 3, stride=1, groups=groups)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EndstopDivideResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDivideResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstopDivideBottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = EndstopDivideBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


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


class AnyStage(Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        for i in range(d):
            block = block_fun(w_in, w_out, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            cx = block_fun.complexity(cx, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return cx


class AnyNet(Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"])
        if cfg.ANYNET.STEM_TYPE == "res_stem_endstop_dilation":
            prev_w = p["stem_w"] * 2
        else:
            prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = AnyHead(prev_w, p["num_classes"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        cx = stem_fun.complexity(cx, 3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = w
        cx = AnyHead.complexity(cx, prev_w, p["num_classes"])
        return cx



