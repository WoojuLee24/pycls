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


class BasicTransformNoBn(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicTransformNoBn, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)

    def forward(self, x):
        x = self.a(x)
        x = self.a_af(x)
        x = self.b(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        return cx


class ResBasicBlockNoBn(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicBlockNoBn, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
        self.f = BasicTransformNoBn(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.proj(x) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx["h"], cx["w"] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class BasicSurroundTransform(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicSurroundTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        # self.a = SurroundDivide(w_in, w_out, 3, stride=stride, groups=1)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        # self.b = conv2d(w_out, w_out, 3)
        self.b = SurroundDivide(w_out, w_out, 3, stride=1, groups=1)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        # x = self.b_bn(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBasicSurroundBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicSurroundBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicSurroundTransform(w_in, w_out, stride, params)
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


class ResBottleneckBlockEndProj(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlockEndproj, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            # self.proj = conv2d(w_in, w_out, 1, stride=stride)
            # self.bn = norm2d(w_out)
            self.proj = EndstoppingDivide(w_in, w_out, 3, stride=stride)
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


class EndstopDilationBottleneckTransformWoBn(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationBottleneckTransformWoBn, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        # self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b = EndstoppingDilation(w_b, w_b, 3, stride=stride, groups=groups)
        # self.b_bn = norm2d(w_b)
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


class EndstopDilationResBottleneckBlockWoBn(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(EndstopDilationResBottleneckBlockWoBn, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = EndstopDilationBottleneckTransformWoBn(w_in, w_out, stride, params)
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


class CompareDilationBottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(CompareDilationBottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = ComparingDilation(w_b, w_b, 3, stride=stride, groups=groups)
        # self.b = EndstoppingDilation(w_b, w_b, 3, stride=stride, groups=groups)
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


class CompareDilationResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(CompareDilationResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = CompareDilationBottleneckTransform(w_in, w_out, stride, params)
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
        cx = CompareDilationBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
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