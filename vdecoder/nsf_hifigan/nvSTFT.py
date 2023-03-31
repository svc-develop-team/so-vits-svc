import math
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import soundfile as sf
import torch.nn.functional as F

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2# check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

class STFT():
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.win_size   = win_size
        self.hop_length = hop_length
        self.fmin     = fmin
        self.fmax     = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}
    
    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels     = self.n_mels
        n_fft      = self.n_fft
        win_size   = self.win_size
        hop_length = self.hop_length
        fmin       = self.fmin
        fmax       = self.fmax
        clip_val   = self.clip_val
        
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        
        mel_basis_key = str(fmax)+'_'+str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
        
        keyshift_key = str(keyshift)+'_'+str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)
        
        pad_left = (win_size_new - hop_length_new) //2
        pad_right = max((win_size_new- hop_length_new + 1) //2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode = mode)
        y = y.squeeze(1)
        
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        # print(111,spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size-resize))
            spec = spec[:, :size, :] * win_size / win_size_new
            
        # print(222,spec)
        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        # print(333,spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # print(444,spec)
        return spec
    
    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

stft = STFT()
