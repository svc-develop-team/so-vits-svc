from modules.F0Predictor.F0Predictor import F0Predictor
import pyworld
import numpy as np
import scipy

class DioF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate

    def resize_f0(self,x, target_len):
        source = np.array(x)
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        return res
    
    def resize_f0_uv(self,x, target_len):
        source = np.array(x)
        vuv_vector = np.zeros_like(x)
        vuv_vector[x > 0.0] = 1.0
        vuv_vector[x < 0.001] = 0.0
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,target_len/len(vuv_vector),order = 0))
        return res,vuv_vector
        
    def compute_f0(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.resize_f0(f0, p_len)

    def compute_f0_uv(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.resize_f0_uv(f0, p_len)