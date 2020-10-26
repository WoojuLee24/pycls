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
import torch.nn as nn
from torch.nn import Module
from pycls.models.endstop_helper import *
from pycls.models.anynet import *


"""
SlowFast style 
"""


class MultiStem(Module):
    """Multi stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out, stem_fun_name):
        super(MultiStem, self).__init__()

        assert(
            len(
                {
                len(w_in),
                len(w_out),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistant"

        self.num_pathways = len(w_in)
        self._construct_stem(w_in, w_out, stem_fun_name)

    def _construct_stem(self, w_in, w_out, stem_fun_name):
        stem_fun = get_stem_fun(stem_fun_name)

        for pathway in range(len(w_in)):
            stem = stem_fun(
                w_in[pathway],
                w_out[pathway],
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
                len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stem_fun_name):
        stem_fun = get_stem_fun(stem_fun_name)
        for pathway in range(len(w_in)):
            cx = stem_fun.complexity(
                cx,
                w_in[pathway],
                w_out[pathway],
            )
        return cx



class FuseStream(Module):
    """
    Fuses the information between the pattern pathway and the shape pathway. Given the
    tensors from pattern pathway and the shape pathway, fuse information between pattern and shape,
    then return the fused tensors from pattern and shape pathway in order.
    """

    def __init__(
            self,
            w_in,
            fusion_conv_channel_ratio,
    ):
        super(FuseStream, self).__init__()
        self.conv_p2s = conv2d(
            w_in[0],
            w_in[0] * fusion_conv_channel_ratio[0],
            kernel_size=[1, 1],
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_p = norm2d(w_in[0] * fusion_conv_channel_ratio[0])
        self.af = activation()
        self.conv_s2p = conv2d(
            w_in[1],
            w_in[1] * fusion_conv_channel_ratio[1],
            kernel_size=[1, 1],
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_s = norm2d(w_in[1] * fusion_conv_channel_ratio[1])

    def forward(self, x):
        x_p = x[0]
        x_s = x[1]
        fuse_s = self.conv_p2s(x_p)
        fuse_s = self.bn(fuse_s)
        fuse_s = self.af(fuse_s)

        fuse_p = self.conv_s2p(x_s)
        fuse_p = self.bn(fuse_p)
        fuse_p = self.af(fuse_p)

        x_p_fuse = torch.cat([x_p, fuse_p], 1)
        x_s_fuse = torch.cat([x_s, fuse_s], 1)
        return [x_p_fuse, x_s_fuse]


class FuseStream(Module):
    """
    Fuses the information between the pattern pathway and the shape pathway. Given the
    tensors from pattern pathway and the shape pathway, fuse information between pattern and shape,
    then return the fused tensors from pattern and shape pathway in order.
    """

    def __init__(
            self,
            w_in,
            fusion_conv_channel_ratio,
    ):
        super(FuseStream, self).__init__()
        self.conv_p2s = conv2d(
            w_in[0],
            w_in[0] * fusion_conv_channel_ratio[0],
            kernel_size=[1, 1],
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_p = norm2d(w_in[0] * fusion_conv_channel_ratio[0])
        self.af = activation()
        self.conv_s2p = conv2d(
            w_in[1],
            w_in[1] * fusion_conv_channel_ratio[1],
            kernel_size=[1, 1],
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_s = norm2d(w_in[1] * fusion_conv_channel_ratio[1])

    def forward(self, x):
        x_p = x[0]
        x_s = x[1]
        fuse_s = self.conv_p2s(x_p)
        fuse_s = self.bn(fuse_s)
        fuse_s = self.af(fuse_s)

        fuse_p = self.conv_s2p(x_s)
        fuse_p = self.bn(fuse_p)
        fuse_p = self.af(fuse_p)

        x_p_fuse = torch.cat([x_p, fuse_p], 1)
        x_s_fuse = torch.cat([x_s, fuse_s], 1)
        return [x_p_fuse, x_s_fuse]


class FuseShapeToPattern(Module):
    """
    Fuses the information between the pattern pathway and the shape pathway. Given the
    tensors from pattern pathway and the shape pathway, fuse information between pattern and shape,
    then return the fused tensors from pattern and shape pathway in order.
    """

    def __init__(
            self,
            w_in,
            fusion_conv_channel_ratio,
    ):
        super(FuseShapeToPattern, self).__init__()

        self.conv_s2p = conv2d(
            w_in,
            w_in * fusion_conv_channel_ratio,
            1,
            stride=1,
            bias=False
        )
        self.bn_p = norm2d(w_in * fusion_conv_channel_ratio)
        self.af = activation()

    def forward(self, x):
        x_p = x[0]
        x_s = x[1]
        fuse_p = self.conv_s2p(x_s)
        fuse_p = self.bn_p(fuse_p)
        fuse_p = self.af(fuse_p)
        x_p_fuse = torch.cat([x_p, fuse_p], 1)

        return [x_p_fuse, x_s]

    @staticmethod
    def complexity(cx, w_in, fusion_conv_channel_ratio):
        cx = conv2d_cx(cx, w_in, w_in*fusion_conv_channel_ratio, 1, stride=1, bias=False)
        cx = norm2d_cx(cx, w_in * fusion_conv_channel_ratio)
        return cx


class AnyMultiStage(Module):
    """AnyMultiNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, num_blocks, block_fun):
        super(AnyMultiStage, self).__init__()

        assert (
                len(
                    {
                        len(w_in),
                        len(w_out),
                        len(stride),
                        len(num_blocks),

                    }
                )
                == 1
        )

        self.num_blocks = num_blocks
        self.num_pathways = len(num_blocks)
        self._construct(
            w_in,
            w_out,
            stride,
            block_fun,
        )

    def _construct(self, w_in, w_out, stride, block_fun):
        for pathway in range(self.num_pathways):
            stride_pathway = stride[pathway]
            for i in range(self.num_blocks[pathway]):
                block = block_fun(w_in[pathway],
                                  w_out[pathway],
                                  stride_pathway,
                                  )
                self.add_module("pathway{}_res{}".format(pathway, i), block)
                stride[pathway], w_in[pathway] = 1, w_out[pathway]

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output

    @staticmethod
    def complexity(cx, w_in, w_out, stride, num_blocks, block_fun):
        for pathway in range(len(w_in)):
            stride_pathway = stride[pathway]
            for i in range(num_blocks[pathway]):
                cx = block_fun.complexity(
                    cx,
                    w_in[pathway],
                    w_out[pathway],
                    stride_pathway,
                )
                stride_pathway, w_in = 1, w_out
        return cx


class AnyMultiNet(Module):
    """AnyMultiNet model."""

    def __init__(self):
        super(AnyMultiNet, self).__init__()
        self.num_pathways = 2
        self._construct_network()
        self.apply(init_weights)

    def _construct_network(self):
        """
        Builds a AnyMulti model. The first pathway is the pattern pathway and the
            second pathway is the shape pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        block_fun = get_block_fun(cfg.ANYMULTINET.BLOCK_TYPE)

        (d2, d3, d4, d5) = cfg.ANYMULTINET.DEPTHS
        w_out_ratio = cfg.ANYMULTINET.BETA_INV // cfg.ANYMULTINET.FUSION_RATIO
        width_per_group = cfg.ANYMULTINET.WIDTH_PER_GROUP

        self.s1 = MultiStem(
            w_in=[3, 3],
            w_out=[width_per_group, width_per_group // cfg.ANYMULTINET.BETA_INV],
            stem_fun_name=cfg.ANYMULTINET.STEM_TYPE
        )

        if cfg.ANYMULTINET.V1:
            self.s1_v1 = EndstopDilationStem(
                w_in=width_per_group // cfg.ANYMULTINET.BETA_INV,
                w_out=width_per_group // cfg.ANYMULTINET.BETA_INV,
            )

        self.s1_fuse = FuseShapeToPattern(
            w_in=width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        self.s2 = AnyMultiStage(
            w_in=[width_per_group + width_per_group // w_out_ratio,
                  width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[4 * width_per_group, 4 * width_per_group//cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d2]*2,
            block_fun=block_fun)

        self.s2_fuse = FuseShapeToPattern(
            w_in=4 * width_per_group//cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        self.s3 = AnyMultiStage(
            w_in=[4 * width_per_group + 4 * width_per_group // w_out_ratio,
                  4 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[8 * width_per_group, 8 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d3] * 2,
            block_fun=block_fun)

        self.s3_fuse = FuseShapeToPattern(
            w_in=8 * width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        self.s4 = AnyMultiStage(
            w_in=[8 * width_per_group + 8 * width_per_group // w_out_ratio,
                  8 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[16 * width_per_group, 16 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d4] * 2,
            block_fun=block_fun)

        self.s4_fuse = FuseShapeToPattern(
            w_in=16 * width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        self.s5 = AnyMultiStage(
            w_in=[16 * width_per_group + 16 * width_per_group // w_out_ratio,
                  16 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[32 * width_per_group, 32 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d5] * 2,
            block_fun=block_fun)


        self.head = BasicAnyHead([32 * width_per_group,
                                  32 * width_per_group//cfg.ANYMULTINET.BETA_INV],
                                 cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        x = [x, x.clone().detach()]
        x = self.s1(x)
        if cfg.ANYMULTINET.V1:
            x[1] = self.s1_v1(x[1])
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)

        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity (if you alter the model, make sure to update)."""
        block_fun = get_block_fun(cfg.ANYMULTINET.BLOCK_TYPE)

        (d2, d3, d4, d5) = cfg.ANYMULTINET.DEPTHS
        w_out_ratio = cfg.ANYMULTINET.BETA_INV // cfg.ANYMULTINET.FUSION_RATIO
        width_per_group = cfg.ANYMULTINET.WIDTH_PER_GROUP
        cx = MultiStem.complexity(
            cx,
            w_in=[3, 3],
            w_out=[width_per_group, width_per_group // cfg.ANYMULTINET.BETA_INV],
            stem_fun_name=cfg.ANYMULTINET.STEM_TYPE
        )
        if cfg.ANYMULTINET.V1:
            cx = EndstopDilationStem.complexity(
                cx,
                w_in=width_per_group // cfg.ANYMULTINET.BETA_INV,
                w_out=width_per_group // cfg.ANYMULTINET.BETA_INV,
            )
        cx = FuseShapeToPattern.complexity(
            cx,
            w_in=width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )
        cx = AnyMultiStage.complexity(
            cx,
            w_in=[width_per_group + width_per_group // w_out_ratio,
                  width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[4 * width_per_group, 4 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d2] * 2,
            block_fun=block_fun)

        cx = FuseShapeToPattern.complexity(
            cx,
            w_in=4 * width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        cx = AnyMultiStage.complexity(
            cx,
            w_in=[4 * width_per_group + 4 * width_per_group // w_out_ratio,
                  4 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[8 * width_per_group, 8 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d3] * 2,
            block_fun=block_fun)

        cx = FuseShapeToPattern.complexity(
            cx,
            w_in=8 * width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        cx = AnyMultiStage.complexity(
            cx,
            w_in=[8 * width_per_group + 8 * width_per_group // w_out_ratio,
                  8 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[16 * width_per_group, 16 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d4] * 2,
            block_fun=block_fun)

        cx = FuseShapeToPattern.complexity(
            cx,
            w_in=16 * width_per_group // cfg.ANYMULTINET.BETA_INV,
            fusion_conv_channel_ratio=cfg.ANYMULTINET.FUSION_RATIO
        )

        cx = AnyMultiStage.complexity(
            cx,
            w_in=[16 * width_per_group + 16 * width_per_group // w_out_ratio,
                  16 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            w_out=[32 * width_per_group, 32 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            stride=cfg.ANYMULTINET.STRIDES[0],
            num_blocks=[d5] * 2,
            block_fun=block_fun)

        cx = BasicAnyHead.complexity(
            cx,
            [32 * width_per_group,
             32 * width_per_group // cfg.ANYMULTINET.BETA_INV],
            cfg.MODEL.NUM_CLASSES)

        return cx









