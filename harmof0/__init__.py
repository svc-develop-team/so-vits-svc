# from https://github.com/wx-wei/harmof0
from .network import HarmoF0
from .pitch_tracker import PitchTracker
import torchaudio
import torch
pit = PitchTracker()


def extract_file_f0(path):
    waveform, sr = torchaudio.load(path)
    time, freq, activation, activation_map = pit.pred(waveform, sr)
    return freq


def extract_wav_f0(wav_1d, sr):
    wav = torch.FloatTensor(wav_1d).unsqueeze(0)
    time, freq, activation, activation_map = pit.pred(wav, sr)
    return freq
