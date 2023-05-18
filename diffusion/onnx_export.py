from diffusion_onnx import GaussianDiffusion
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from wavenet import WaveNet
import torch.nn.functional as F
import diffusion

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                128,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden)
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args


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
        self.decoder = GaussianDiffusion(out_dims, n_layers, n_chans, n_hidden)
        self.hidden_size = n_hidden
        self.speaker_map = torch.zeros((self.n_spk,1,1,n_hidden))
    
        

    def forward(self, units, mel2ph, f0, volume, g = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        decoder_inp = F.pad(units, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units.shape[-1]])
        units = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        x = self.unit_embed(units) + self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()) + self.volume_embed(volume.unsqueeze(-1))

        if self.n_spk is not None and self.n_spk > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
            x = x.transpose(1, 2) + g
            return x
        else:
            return x.transpose(1, 2)
        

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

    def OnnxExport(self, project_name=None, init_noise=None, export_encoder=True, export_denoise=True, export_pred=True, export_after=True):
        hubert_hidden_size = 768
        n_frames = 100
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        spk_mix = spk_mix.repeat(n_frames, 1)
        orgouttt = self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        outtt = self.forward(hubert, mel2ph, f0, volume, spk_mix)
        if export_encoder:
            torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1],
                    "spk_mix": [0],
                },
                opset_version=16
            )
        
        self.decoder.OnnxExport(project_name, init_noise=init_noise, export_denoise=export_denoise, export_pred=export_pred, export_after=export_after)

    def ExportOnnx(self, project_name=None):
        hubert_hidden_size = 768
        n_frames = 100
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        orgouttt = self.orgforward(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        outtt = self.forward(hubert, mel2ph, f0, volume, spk_mix)

        torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1]
                },
                opset_version=16
            )

        condition = torch.randn(1,self.decoder.n_hidden,n_frames)
        noise = torch.randn((1, 1, self.decoder.mel_bins, condition.shape[2]), dtype=torch.float32)
        pndm_speedup = torch.LongTensor([100])
        K_steps = torch.LongTensor([1000])
        self.decoder = torch.jit.script(self.decoder)
        self.decoder(condition, noise, pndm_speedup, K_steps)

        torch.onnx.export(
                self.decoder,
                (condition, noise, pndm_speedup, K_steps),
                f"{project_name}_diffusion.onnx",
                input_names=["condition", "noise", "pndm_speedup", "K_steps"],
                output_names=["mel"],
                dynamic_axes={
                    "condition": [2],
                    "noise": [3],
                },
                opset_version=16
            )


if __name__ == "__main__":
    project_name = "dddsp"
    model_path = f'{project_name}/model_500000.pt'

    model, _ = load_model_vocoder(model_path)

    # 分开Diffusion导出（需要使用MoeSS/MoeVoiceStudio或者自己编写Pndm/Dpm采样）
    model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

    # 合并Diffusion导出（Encoder和Diffusion分开，直接将Encoder的结果和初始噪声输入Diffusion即可）
    # model.ExportOnnx(project_name)

