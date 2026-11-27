"""
PMNetSlim: PMNet with a channel-width multiplier and a configurable number of input channels.

Adapted from PMNet (Ju-Hyung Lee et al., https://github.com/abman23/PMNet, MIT license): pmnet_v1.py and pmnet_v3.py,
which differ only in the output stride (16 and 8) and in one decoder layer. This file merges them, replaces the
hard-coded channel numbers by multiples of `width`, and takes the number of input channels as an argument.
If you redistribute this file, keep the MIT license notice of the PMNet repository with it.

    width = 1.0   same network as PMNet (33.4 M parameters with 5 input channels, blocks 3 3 27 3)
    width = 0.5   every layer has half the channels: about 8.4 M parameters and a quarter of the computation

The structure (encoder with atrous convolutions, ASPP, decoder with skip connections) is unchanged.
Use it through train_pmnet.py:  --width 0.5
"""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """Convolution, batch norm, ReLU: the basic building block."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU, self).__init__()
        self.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1))
        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """ResNet bottleneck block built from _ConvBnReLU."""

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False) if downsample else nn.Identity()

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)
        for i in range(n_layers):          # downsampling only in the first block
            self.add_module("block{}".format(i + 1), _Bottleneck(
                in_ch=(in_ch if i == 0 else out_ch), out_ch=out_ch, stride=(stride if i == 0 else 1),
                dilation=dilation * multi_grids[i], downsample=(i == 0)))


class _Stem(nn.Sequential):
    """First layer: 7x7 convolution with stride 2, then max pooling (kernel 2, stride 2, padding 1, as in PMNet)."""

    def __init__(self, out_ch, in_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(2, 2, 1, ceil_mode=True))


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.conv(self.pool(x))
        return F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous spatial pyramid pooling: parallel convolutions with different dilation rates."""

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module("c{}".format(i + 1), _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate))
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


def ConRu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding), nn.ReLU(inplace=True))


def ConRuT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
                         nn.ReLU(inplace=True))


class PMNetSlim(nn.Module):

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, in_channels=5, width=1.0, bn_momentum=0.1):
        super(PMNetSlim, self).__init__()
        if output_stride == 8:          # "v3"
            s, d = [1, 2, 1, 1], [1, 1, 2, 4]
        elif output_stride == 16:       # "v1"
            s, d = [1, 2, 2, 1], [1, 1, 1, 2]
        else:
            raise ValueError("output_stride must be 8 or 16")

        def c(n):                       # scaled channel number, a multiple of 4 (the bottleneck divides by 4)
            return max(8, int(round(n * width / 4.0)) * 4)

        C64, C128, C256, C512, C1024 = c(64), c(128), c(256), c(512), c(1024)

        # Encoder
        self.layer1 = _Stem(C64, in_channels)
        self.layer2 = _ResLayer(n_blocks[0], C64, C256, s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], C256, C512, s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], C512, C512, s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], C512, C1024, s[3], d[3], multi_grids)
        self.aspp = _ASPP(C1024, C256, atrous_rates)
        self.add_module("fc1", _ConvBnReLU(C256 * (len(atrous_rates) + 2), C512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(C256, C256, 1, 1, 0, 1)

        # Decoder
        self.conv_up5 = ConRu(C512, C512, 3, 1)
        self.conv_up4 = (ConRu if output_stride == 8 else ConRuT)(C512 + C512, C512, 3, 1)
        self.conv_up3 = ConRuT(C512 + C512, C256, 3, 1)
        self.conv_up2 = ConRu(C256 + C256, C256, 3, 1)
        self.conv_up1 = ConRu(C256 + C256, C256, 3, 1)
        self.conv_up0 = ConRu(C256 + C64, C128, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(C128 + in_channels, C64, kernel_size=3, padding=1), nn.BatchNorm2d(C64), nn.ReLU(),
            nn.Conv2d(C64, C64, kernel_size=3, padding=1), nn.BatchNorm2d(C64), nn.ReLU(),
            nn.Conv2d(C64, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        return self.conv_up00(xup0)