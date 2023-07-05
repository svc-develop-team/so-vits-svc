# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)
        self.conv_transpose1d_block = None
        if C is not None:
            self.conv_transpose1d_block = [nn.ConvTranspose1d(C,
                                                            C,
                                                            kernel_size=self.kernel_size,
                                                            stride=self.stride, 
                                                            groups=C, 
                                                            bias=False
                                                            ),]
            self.conv_transpose1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1).clone())
            self.conv_transpose1d_block[0].requires_grad_(False)
            
            

    # x: [B, C, T]
    def forward(self, x, C=None):
        if self.conv_transpose1d_block[0].weight.device != x.device:
            self.conv_transpose1d_block[0] = self.conv_transpose1d_block[0].to(x.device)
        if self.conv_transpose1d_block is None:
            if C is None:
                _, C, _ = x.shape
            # print("snake.conv_t.in:",x.shape)
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
            # print("snake.conv_t.out:",x.shape)
            x = x[..., self.pad_left:-self.pad_right]
        else:
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * self.conv_transpose1d_block[0](x)
            x = x[..., self.pad_left:-self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size,
                                       C=C)


    def forward(self, x):
        xx = self.lowpass(x)

        return xx