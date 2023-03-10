from math import sin
import torch
from torch._C import has_openmp
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .layers import MRDConv, FRDConv, WaveformToLogSpecgram

def dila_conv_block( 
    in_channel, out_channel, 
    bins_per_octave,
    n_har,
    dilation_mode,
    dilation_rate,
    dil_kernel_size,
    kernel_size = [1,3],
    padding = [0,1],
):

    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channel)

    # dilation mode: 'log_scale', 'fixed'
    if(dilation_mode == 'log_scale'):
        a = np.log(np.arange(1, n_har + 1))/np.log(2**(1.0/bins_per_octave))
        dilation_list = a.round().astype(np.int)
        conv_log_dil = MRDConv(out_channel, out_channel, dilation_list)
        return nn.Sequential(
            conv,nn.ReLU(),
            conv_log_dil,nn.ReLU(),
            batch_norm,
            # pool
        )
    elif(dilation_mode == 'fixed_causal'):
        dilation_list = np.array([i * dil_kernel_size[1] for i in range(dil_kernel_size[1])])
        causal_conv = FRDConv(out_channel, out_channel, dil_kernel_size, dilation=[1, dilation_rate])
        return nn.Sequential(
            conv,nn.ReLU(),
            causal_conv,nn.ReLU(),
            batch_norm,
            # pool
        )
    elif(dilation_mode == 'fixed'):
        conv_dil = nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size, padding='same', dilation=[1, dilation_rate])
        
        return nn.Sequential(
            conv,nn.ReLU(),
            conv_dil,nn.ReLU(),
            batch_norm,
            # pool
        )
    else:
        assert False, "unknown dilation type: " + dilation_mode


class HarmoF0(nn.Module):
    def __init__(self, 
            sample_rate=16000, 
            n_freq=512, 
            n_har=12, 
            bins_per_octave=12 * 4, 
            dilation_modes=['log_scale', 'fixed', 'fixed', 'fixed'],
            dilation_rates=[48, 48, 48, 48],
            logspecgram_type='logharmgram',
            channels=[32, 64, 128, 128],
            fmin=27.5,
            freq_bins=88 * 4,
            dil_kernel_sizes= [[1, 3], [1,3], [1,3], [1,3]],
        ):
        super().__init__()
        self.logspecgram_type = logspecgram_type

        n_fft = n_freq * 2
        self.n_freq = n_freq
        self.freq_bins = freq_bins
        
        self.waveform_to_logspecgram = WaveformToLogSpecgram(sample_rate, n_fft, fmin, bins_per_octave, freq_bins, n_freq, logspecgram_type) #, device

        bins = bins_per_octave

        # [b x 1 x T x 88*8] => [b x 32 x T x 88*4]
        self.block_1 = dila_conv_block(1, channels[0], bins, n_har=n_har, dilation_mode=dilation_modes[0], dilation_rate=dilation_rates[0], dil_kernel_size=dil_kernel_sizes[0], kernel_size=[3, 3], padding=[1,1])
        
        bins = bins // 2
        # => [b x 64 x T x 88*4]
        self.block_2 = dila_conv_block(channels[0], channels[1], bins, 3, dilation_mode=dilation_modes[1], dilation_rate=dilation_rates[1], dil_kernel_size=dil_kernel_sizes[1], kernel_size=[3, 3], padding=[1,1])
        # => [b x 128 x T x 88*4]
        self.block_3 = dila_conv_block(channels[1], channels[2], bins, 3, dilation_mode=dilation_modes[2], dilation_rate=dilation_rates[2], dil_kernel_size=dil_kernel_sizes[2], kernel_size=[3, 3], padding=[1,1])
        # => [b x 128 x T x 88*4]
        self.block_4 = dila_conv_block(channels[2], channels[3], bins, 3, dilation_mode=dilation_modes[3], dilation_rate=dilation_rates[3], dil_kernel_size=dil_kernel_sizes[3], kernel_size=[3, 3], padding=[1,1])

        self.conv_5 = nn.Conv2d(channels[3], channels[3]//2, kernel_size=[1,1])
        self.conv_6 = nn.Conv2d(channels[3]//2, 1, kernel_size=[1,1])

    def forward(self, waveforms):
        # input: [b x num_frames x frame_len]
        # output: [b x num_frames x 352], [b x num_frames x 352]

        specgram = self.waveform_to_logspecgram(waveforms).float()
        # => [b x 1 x num_frames x n_bins]
        x = specgram[None, :]

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        # [b x 128 x T x 352] => [b x 64 x T x 352]
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x, dim=1)
        # x = torch.clip(x, 1e-4, 1 - 1e-4)
        # => [num_frames x n_bins]
        return x, specgram


