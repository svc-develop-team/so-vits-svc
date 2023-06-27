import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .wavenet import WaveNet


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None
        ):
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded diffusion model, sampler is {args.infer.method}, speedup: {args.infer.speedup} ')
    return model, vocoder, args


class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        
        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max>0 and k_step_max<self.timesteps else self.timesteps

        self.n_hidden = n_hidden
        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden),timesteps=self.timesteps,k_step=self.k_step_max, out_dims=out_dims)
        self.input_channel = input_channel
    
    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                spk_embed_mix = torch.zeros((1,1,self.hidden_size))
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                x = x + spk_embed_mix
            else:
                x = x + self.spk_embed(spk_id - 1)
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()
        return x.transpose(1, 2)

    def init_spkmix(self, n_spk):
        self.speaker_map = torch.zeros((n_spk,1,1,self.n_hidden))
        hubert_hidden_size = self.input_channel
        n_frames = 10
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spks = {}
        for i in range(n_spk):
            spks.update({i:1.0/float(self.n_spk)})
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)

    def forward(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        if not self.training and gt_spec is not None and k_step>self.k_step_max:
            raise Exception("The shallow diffusion k_step is greater than the maximum diffusion k_step(k_step_max)!")

        if not self.training and gt_spec is None and self.k_step_max!=self.timesteps:
            raise Exception("This model can only be used for shallow diffusion and can not infer alone!")

        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                if spk_id.shape[1] > 1:
                    g = spk_id.reshape((spk_id.shape[0], spk_id.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
                    g = g * self.speaker_map  # [N, S, B, 1, H]
                    g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
                    g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
                    x = x + g
                else:
                    x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
    
        return x

