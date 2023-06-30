import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from vdecoder.nsf_hifigan.models import load_model
from vdecoder.nsf_hifigan.nvSTFT import STFT


class Enhancer:
    def __init__(self, enhancer_type, enhancer_ckpt, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if enhancer_type == 'nsf-hifigan':
            self.enhancer = NsfHifiGAN(enhancer_ckpt, device=self.device)
        else:
            raise ValueError(f" [x] Unknown enhancer: {enhancer_type}")
        
        self.resample_kernel = {}
        self.enhancer_sample_rate = self.enhancer.sample_rate()
        self.enhancer_hop_size = self.enhancer.hop_size()
        
    def enhance(self,
                audio, # 1, T
                sample_rate,
                f0, # 1, n_frames, 1
                hop_size,
                adaptive_key = 0,
                silence_front = 0
                ):
        # enhancer start time 
        start_frame = int(silence_front * sample_rate / hop_size)
        real_silence_front = start_frame * hop_size / sample_rate
        audio = audio[:, int(np.round(real_silence_front * sample_rate)) : ]
        f0 = f0[: , start_frame :, :]
        
        # adaptive parameters
        adaptive_factor = 2 ** ( -adaptive_key / 12)
        adaptive_sample_rate = 100 * int(np.round(self.enhancer_sample_rate / adaptive_factor / 100))
        real_factor = self.enhancer_sample_rate / adaptive_sample_rate
        
        # resample the ddsp output
        if sample_rate == adaptive_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate) + str(adaptive_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, adaptive_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        n_frames = int(audio_res.size(-1) // self.enhancer_hop_size + 1)
        
        # resample f0
        f0_np = f0.squeeze(0).squeeze(-1).cpu().numpy()
        f0_np *= real_factor
        time_org = (hop_size / sample_rate) * np.arange(len(f0_np)) / real_factor
        time_frame = (self.enhancer_hop_size / self.enhancer_sample_rate) * np.arange(n_frames)
        f0_res = np.interp(time_frame, time_org, f0_np, left=f0_np[0], right=f0_np[-1])
        f0_res = torch.from_numpy(f0_res).unsqueeze(0).float().to(self.device) # 1, n_frames

        # enhance
        enhanced_audio, enhancer_sample_rate = self.enhancer(audio_res, f0_res)
        
        # resample the enhanced output
        if adaptive_factor != 0:
            key_str = str(adaptive_sample_rate) + str(enhancer_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(adaptive_sample_rate, enhancer_sample_rate, lowpass_filter_width = 128).to(self.device)
            enhanced_audio =  self.resample_kernel[key_str](enhanced_audio)
        
        # pad the silence frames
        if start_frame > 0:
            enhanced_audio = F.pad(enhanced_audio, (int(np.round(enhancer_sample_rate * real_silence_front)), 0))
            
        return enhanced_audio, enhancer_sample_rate
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print('| Load HifiGAN: ', model_path)
        self.model, self.h = load_model(model_path, device=self.device)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
        
    def forward(self, audio, f0):
        stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
        with torch.no_grad():
            mel = stft.get_mel(audio)
            enhanced_audio = self.model(mel, f0[:,:mel.size(-1)]).view(-1)
            return enhanced_audio, self.h.sampling_rate