import io
import logging
import os

import librosa
import numpy as np
import parselmouth
import soundfile
import torch
import torchaudio

import utils
from inference import slicer
from models import SynthesizerTrn

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res

def get_f0(x, p_len,f0_up_key=0):

    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class VitsSvc(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SVCVITS = None
        self.hps = None
        self.speakers = None
        self.hubert_soft = utils.get_hubert_model()

    def set_device(self, device):
        self.device = torch.device(device)
        self.hubert_soft.to(self.device)
        if self.SVCVITS is not None:
            self.SVCVITS.to(self.device)

    def loadCheckpoint(self, path):
        self.hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        self.SVCVITS = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", self.SVCVITS, None)
        _ = self.SVCVITS.eval().to(self.device)
        self.speakers = self.hps.spk

    def get_units(self, source, sr):
        source = source.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            units = self.hubert_soft.units(source)
            return units


    def get_unit_pitch(self, in_path, tran):
        source, sr = torchaudio.load(in_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0]*2, tran)
        return soft, f0

    def infer(self, speaker_id, tran, raw_path):
        speaker_id = self.speakers[speaker_id]
        sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.device)
        stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            audio,_ = self.SVCVITS.infer(x_tst, f0=f0, g=sid)[0,0].data.float()
        return audio, audio.shape[-1]

    def inference(self,srcaudio,chara,tran,slice_db):
        sampling_rate, audio = srcaudio
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        soundfile.write("tmpwav.wav", audio, 16000, format="wav")
        chunks = slicer.cut("tmpwav.wav", db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio("tmpwav.wav", chunks)
        audio = []
        for (slice_tag, data) in audio_data:
            length = int(np.ceil(len(data) / audio_sr * self.hps.data.sampling_rate))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = self.infer(chara, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))
        audio = (np.array(audio) * 32768.0).astype('int16')
        return (self.hps.data.sampling_rate,audio)
