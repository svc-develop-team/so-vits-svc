import io
import logging
import time
from pathlib import Path
import os

import librosa
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

model_path = "logs/32k/G_174000-Copy1.pth"
config_path = "configs/config.json"
svc_model = Svc(model_path, config_path)
infer_tool.mkdir(["raw", "results"])

# 支持多个wav文件，放在raw文件夹下
clean_names = ["君の知らない物語-src"]
trans = [-5]  # 音高调整，支持正负（半音）
spk_list = ['yunhao']  # 每次同时合成多语者音色
slice_db = -40  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
wav_format = 'flac'  # 音频输出格式
clip = 0     # 音频自动切片，0为不切片，单位为秒/s
cpu = False  # 是否使用CPU推断，若使用请改为True

if cpu : os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
svc_model = Svc(model_path, config_path)
infer_tool.mkdir(["raw", "results"])
infer_tool.fill_a_to_b(trans, clean_names)
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"raw/{clean_name}"
    if "." not in raw_audio_path:
        raw_audio_path += ".wav"
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix('.wav')
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
    per_size = clip*audio_sr
    for spk in spk_list:
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(_audio))
                continue
            if per_size != 0:
                datas = infer_tool.split_list_by_n(data, per_size)
            else:
                datas = [data]
            for dat in datas:
                if clip!=0: print(f'  #=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
                _audio = out_audio.cpu().numpy()
                audio.extend(list(_audio))

        res_path = f'./results/{clean_name}_{tran}key_{spk}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
