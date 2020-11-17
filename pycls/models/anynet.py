#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

from pycls.core.config import cfg
import torch
from torch.nn import Module
from pycls.models.stem_helper import *
from pycls.models.bottleneck_helper import *

def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "res_stem_compare": ResStemCompare,
        "res_stem_compare_dilation_separation_entire": ResStemCompareDilationSeparationEntire,
        "res_stem_compare_dilation_separation_bn_entire": ResStemCompareDilationSeparationBnEntire,
        "simple_stem_in": SimpleStem,
        "res_stem_endstop_dilation": ResStemEndstopDilation,
        "endstop_dilation_stem": EndstopDilationStem,
        "res_stem_endstop_dilation_separation_entire": ResStemEndstopDilationSeparationEntire,
        "res_stem_endstop_dilation_separation_bn_entire": ResStemEndstopDilationSeparationBnEntire,
        "res_stem_endstop_dilation_prelu_separation_entire": ResStemEndstopDilationPreluSeparationEntire,
        "res_stem_endstop_dilation_prelu_separation_entire": ResStemEndstopDilationPreluSeparationBnEntire,
        "res_stem_endstop_dilation_fc_entire": ResStemEndstopDilationFcEntire,
        "res_stem_fix_sm_entire": ResStemFixSMEntire,
        "res_stem_endstop_divide_separation": ResStemEndstopDivideSeparation,
        "res_stem_endstop_divide_separation_entire": ResStemEndstopDivideSeparationEntire,
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
        "res_bottleneck_block_end_proj": ResBottleneckBlockEndProj,
        "endstop_dilation_res_bottleneck_block": EndstopDilationResBottleneckBlock,
        "compare_dilation_res_bottleneck_block": CompareDilationResBottleneckBlock,
        "endstop_divide_res_bottleneck_block": EndstopDivideResBottleneckBlock,
        "endstop_dilation_prelu_res_bottleneck_block": EndstopDilationPReLUResBottleneckBlock,
        "endstop_dilation_with_conv_res_bottleneck_block": EndstopDilationWithConvResBottleneckBlock,
        "basic_res_bottleneck_block": BasicResBottleneckBlock,
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
        if (cfg.ANYNET.STEM_TYPE == "res_stem_endstop_dilation") or (cfg.ANYNET.STEM_TYPE == "res_stem_endstop_divide_separation")\
                or (cfg.ANYNET.STEM_TYPE == "res_stem_compare"):
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



