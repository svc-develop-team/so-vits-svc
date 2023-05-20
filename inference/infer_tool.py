import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from inference import slicer
import gc

import librosa
import numpy as np
# import onnxruntime
import soundfile
import torch
import torchaudio

import cluster
import utils
from models import SynthesizerTrn

from diffusion.unit2mel import load_model_vocoder
import yaml

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()

def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr
    
def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]


class F0FilterException(Exception):
    pass

class Svc(object):
    def __init__(self, net_g_path, config_path,
                 device=None,
                 cluster_model_path="logs/44k/kmeans_10000.pt",
                 nsf_hifigan_enhance = False,
                 diffusion_model_path="logs/44k/diffusion/model_0.pt",
                 diffusion_config_path="configs/diffusion.yaml",
                 shallow_diffusion = False,
                 only_diffusion = False,
                 ):
        self.net_g_path = net_g_path
        self.only_diffusion = only_diffusion
        self.shallow_diffusion = shallow_diffusion
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.net_g_ms = None
        if not self.only_diffusion:
            self.hps_ms = utils.get_hparams_from_file(config_path)
            self.target_sample = self.hps_ms.data.sampling_rate
            self.hop_size = self.hps_ms.data.hop_length
            self.spk2id = self.hps_ms.spk
            try:
                self.speech_encoder = self.hps_ms.model.speech_encoder
            except Exception as e:
                self.speech_encoder = 'vec768l12'

        self.nsf_hifigan_enhance = nsf_hifigan_enhance
        if self.shallow_diffusion or self.only_diffusion:
            if os.path.exists(diffusion_model_path) and os.path.exists(diffusion_model_path):
                self.diffusion_model,self.vocoder,self.diffusion_args = load_model_vocoder(diffusion_model_path,self.dev,config_path=diffusion_config_path)
                if self.only_diffusion:
                    self.target_sample = self.diffusion_args.data.sampling_rate
                    self.hop_size = self.diffusion_args.data.block_size
                    self.spk2id = self.diffusion_args.spk
                    self.speech_encoder = self.diffusion_args.data.encoder
            else:
                print("No diffusion model or config found. Shallow diffusion mode will False")
                self.shallow_diffusion = self.only_diffusion = False
                
        # load hubert and model
        if not self.only_diffusion:
            self.load_model()
            self.hubert_model = utils.get_speech_encoder(self.speech_encoder,device=self.dev)
            self.volume_extractor = utils.Volume_Extractor(self.hop_size)
        else:
            self.hubert_model = utils.get_speech_encoder(self.diffusion_args.data.encoder,device=self.dev)
            self.volume_extractor = utils.Volume_Extractor(self.diffusion_args.data.block_size)
            
        if os.path.exists(cluster_model_path):
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        if self.shallow_diffusion : self.nsf_hifigan_enhance = False
        if self.nsf_hifigan_enhance:
            from modules.enhancer import Enhancer
            self.enhancer = Enhancer('nsf-hifigan', 'pretrain/nsf_hifigan/model',device=self.dev)
            
    def load_model(self):
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)



    def get_unit_f0(self, wav, tran, cluster_infer_ratio, speaker, f0_filter ,f0_predictor,cr_threshold=0.05):

        f0_predictor_object = utils.get_f0_predictor(f0_predictor,hop_length=self.hop_size,sampling_rate=self.target_sample,device=self.dev,threshold=cr_threshold)
        
        f0, uv = f0_predictor_object.compute_f0_uv(wav)
        if f0_filter and sum(f0) == 0:
            raise F0FilterException("No voice detected")
        f0 = torch.FloatTensor(f0).to(self.dev)
        uv = torch.FloatTensor(uv).to(self.dev)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav16k = librosa.resample(wav, orig_sr=self.target_sample, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(self.dev)
        c = self.hubert_model.encoder(wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1])

        if cluster_infer_ratio !=0:
            cluster_c = cluster.get_cluster_center_result(self.cluster_model, c.cpu().numpy().T, speaker).T
            cluster_c = torch.FloatTensor(cluster_c).to(self.dev)
            c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

    def infer(self, speaker, tran, raw_path,
              cluster_infer_ratio=0,
              auto_predict_f0=False,
              noice_scale=0.4,
              f0_filter=False,
              f0_predictor='pm',
              enhancer_adaptive_key = 0,
              cr_threshold = 0.05,
              k_step = 100
              ):

        speaker_id = self.spk2id.get(speaker)
        if not speaker_id and type(speaker) is int:
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        wav, sr = librosa.load(raw_path, sr=self.target_sample)
        c, f0, uv = self.get_unit_f0(wav, tran, cluster_infer_ratio, speaker, f0_filter,f0_predictor,cr_threshold=cr_threshold)
        if "half" in self.net_g_path and torch.cuda.is_available():
            c = c.half()
        with torch.no_grad():
            start = time.time()
            if not self.only_diffusion:
                audio,f0 = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noice_scale=noice_scale)
                audio = audio[0,0].data.float()
                if self.shallow_diffusion:
                    audio_mel = self.vocoder.extract(audio[None,:],self.target_sample)
                    vol = self.volume_extractor.extract(audio[None,:])[None,:,None].to(self.dev)
                    f0 = f0[:,:,None]
                    c = c.transpose(-1,-2)
                    audio_mel = self.diffusion_model(
                    c, 
                    f0, 
                    vol, 
                    spk_id = sid, 
                    spk_mix_dict = None,
                    gt_spec=audio_mel,
                    infer=True, 
                    infer_speedup=self.diffusion_args.infer.speedup, 
                    method=self.diffusion_args.infer.method,
                    k_step=k_step)
                    audio = self.vocoder.infer(audio_mel, f0).squeeze()
            else:
                wav = torch.FloatTensor(wav).to(self.dev)
                vol = self.volume_extractor.extract(wav[None,:])[None,:,None].to(self.dev)
                c = c.transpose(-1,-2)
                f0 = f0[:,:,None]
                audio_mel = self.diffusion_model(
                    c, 
                    f0, 
                    vol, 
                    spk_id = sid, 
                    spk_mix_dict = None,
                    gt_spec=None,
                    infer=True,
                    infer_speedup=self.diffusion_args.infer.speedup, 
                    method=self.diffusion_args.infer.method,
                    k_step=k_step)
                audio = self.vocoder.infer(audio_mel, f0).squeeze()
            if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance(
                                    audio[None,:], 
                                    self.target_sample, 
                                    f0[:,:,None], 
                                    self.hps_ms.data.hop_length, 
                                    adaptive_key = enhancer_adaptive_key)
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1]

    def clear_empty(self):
        # clean up vram
        torch.cuda.empty_cache()

    def unload_model(self):
        # unload model
        self.net_g_ms = self.net_g_ms.to("cpu")
        del self.net_g_ms
        if hasattr(self,"enhancer"): 
            self.enhancer.enhancer = self.enhancer.enhancer.to("cpu")
            del self.enhancer.enhancer
            del self.enhancer
        gc.collect()

    def slice_inference(self,
                        raw_audio_path,
                        spk,
                        tran,
                        slice_db,
                        cluster_infer_ratio,
                        auto_predict_f0,
                        noice_scale,
                        pad_seconds=0.5,
                        clip_seconds=0,
                        lg_num=0,
                        lgr_num =0.75,
                        f0_predictor='pm',
                        enhancer_adaptive_key = 0,
                        cr_threshold = 0.05,
                        k_step = 100
                        ):
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip_seconds*audio_sr)
        lg_size = int(lg_num*audio_sr)
        lg_size_r = int(lg_size*lgr_num)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0
        
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size,lg_size)
            else:
                datas = [data]
            for k,dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if clip_seconds!=0 else length
                if clip_seconds!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                # padd
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr = self.infer(spk, tran, raw_path,
                                                    cluster_infer_ratio=cluster_infer_ratio,
                                                    auto_predict_f0=auto_predict_f0,
                                                    noice_scale=noice_scale,
                                                    f0_predictor = f0_predictor,
                                                    enhancer_adaptive_key = enhancer_adaptive_key,
                                                    cr_threshold = cr_threshold,
                                                    k_step = k_step
                                                    )
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                if lg_size!=0 and k!=0:
                    lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr_num != 1 else audio[-lg_size:]
                    lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr_num != 1 else _audio[0:lg_size]
                    lg_pre = lg1*(1-lg)+lg2*lg
                    audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr_num != 1 else audio[0:-lg_size]
                    audio.extend(lg_pre)
                    _audio = _audio[lg_size_c_l+lg_size_r:] if lgr_num != 1 else _audio[lg_size:]
                audio.extend(list(_audio))
        return np.array(audio)

class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # chunk length
        self.pre_len = 3840  # cross fade length, multiples of 640

    # Input and output are 1-dimensional numpy waveform arrays

    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path,
                cluster_infer_ratio=0,
                auto_predict_f0=False,
                noice_scale=0.4,
                f0_filter=False):

        import maad
        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)
            
            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav,
                                        cluster_infer_ratio=cluster_infer_ratio,
                                        auto_predict_f0=auto_predict_f0,
                                        noice_scale=noice_scale,
                                        f0_filter=f0_filter)

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]
            