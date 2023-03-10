import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text, transform

# import h5py


"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        c = torch.load(filename + ".soft.pt").squeeze(0)
        c = torch.repeat_interleave(c, repeats=2, dim=1)

        f0 = np.load(filename + ".f0.npy")
        f0 = torch.FloatTensor(f0)
        lmin = min(c.size(-1), spec.size(-1), f0.shape[0])
        assert abs(c.size(-1) - spec.size(-1)) < 4, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(lmin - spec.size(-1)) < 4, (c.size(-1), spec.size(-1), f0.shape)
        assert abs(lmin - c.size(-1)) < 4, (c.size(-1), spec.size(-1), f0.shape)
        spec, c, f0 = spec[:, :lmin], c[:, :lmin], f0[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        _spec, _c, _audio_norm, _f0 = spec, c, audio_norm, f0
        while spec.size(-1) < self.spec_len:
            spec = torch.cat((spec, _spec), -1)
            c = torch.cat((c, _c), -1)
            f0 = torch.cat((f0, _f0), -1)
            audio_norm = torch.cat((audio_norm, _audio_norm), -1)
        start = random.randint(0, spec.size(-1) - self.spec_len)
        end = start + self.spec_len
        spec = spec[:, start:end]
        c = c[:, start:end]
        f0 = f0[start:end]
        audio_norm = audio_norm[:, start * self.hop_length:end * self.hop_length]

        return c, f0, spec, audio_norm, spk

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)


class EvalDataLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.audiopaths = self.audiopaths[:5]
        self.spk_map = hparams.spk


    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        c = torch.load(filename + ".soft.pt").squeeze(0)

        c = torch.repeat_interleave(c, repeats=2, dim=1)

        f0 = np.load(filename + ".f0.npy")
        f0 = torch.FloatTensor(f0)
        lmin = min(c.size(-1), spec.size(-1), f0.shape[0])
        assert abs(c.size(-1) - spec.size(-1)) < 4, (c.size(-1), spec.size(-1), f0.shape)
        assert abs(f0.shape[0] - spec.shape[-1]) < 4, (c.size(-1), spec.size(-1), f0.shape)
        spec, c, f0 = spec[:, :lmin], c[:, :lmin], f0[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]

        return c, f0, spec, audio_norm, spk

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)

