import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from .diffusion import GaussianDiffusion
from .wavenet import WaveNet
from .vocoder import Vocoder

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
    if config_path is None: config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else: config_file = config_path
    
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
                args.model.n_hidden)
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
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
            n_hidden=256):
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
            
        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
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
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
    
        return x

