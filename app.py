import gradio as gr
import os
os.system('cd monotonic_align && python setup.py build_ext --inplace && cd ..')

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

import librosa
import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
def resize2d(source, target_len):
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    return np.nan_to_num(target)
def convert_wav_22050_to_f0(audio):
    tmp = librosa.pyin(audio,
                fmin=librosa.note_to_hz('C0'),
                fmax=librosa.note_to_hz('C7'),
                frame_length=1780)[0]
    f0 = np.zeros_like(tmp)
    f0[tmp>0] = tmp[tmp>0]
    return f0

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    print(text_norm.shape)
    return text_norm


hps = utils.get_hparams_from_file("configs/ljs_base.json")
hps_ms = utils.get_hparams_from_file("configs/vctk_base.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)

import numpy as np

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

_ = utils.load_checkpoint("G_312000.pth", net_g_ms, None)

def vc_fn(input_audio,vc_transform):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 30:
        return "Error: Audio is too long", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    audio22050 = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    f0 = convert_wav_22050_to_f0(audio22050)

    source = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    print(source.shape)
    with torch.inference_mode():
        units = hubert.units(source)
        soft = units.squeeze(0).numpy()
        print(sampling_rate)
        f0 = resize2d(f0, len(soft[:, 0])) * vc_transform
        soft[:, 0] = f0 / 10
    sid = torch.LongTensor([0])
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths,sid=sid, noise_scale=0.1, noise_scale_w=0.1, length_scale=1)[0][
            0, 0].data.float().numpy()

    return "Success", (hps.data.sampling_rate, audio)



app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            vc_input3 = gr.Audio(label="Input Audio (30s limitation)")
            vc_transform = gr.Number(label="transform",value=1.0)
            vc_submit = gr.Button("Convert", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [ vc_input3,vc_transform], [vc_output1, vc_output2])

    app.launch()