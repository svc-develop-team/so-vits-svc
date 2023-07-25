import gc
import hashlib
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path

import librosa
import numpy as np

# import onnxruntime
import soundfile
import torch
import torchaudio

import cluster
import utils
from diffusion.unit2mel import load_model_vocoder
from inference import slicer
from models import SynthesizerTrn

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
                 spk_mix_enable = False,
                 feature_retrieval = False
                 ):
        self.net_g_path = net_g_path
        self.only_diffusion = only_diffusion
        self.shallow_diffusion = shallow_diffusion
        self.feature_retrieval = feature_retrieval
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.net_g_ms = None
        if not self.only_diffusion:
            self.hps_ms = utils.get_hparams_from_file(config_path,True)
            self.target_sample = self.hps_ms.data.sampling_rate
            self.hop_size = self.hps_ms.data.hop_length
            self.spk2id = self.hps_ms.spk
            self.unit_interpolate_mode = self.hps_ms.data.unit_interpolate_mode if self.hps_ms.data.unit_interpolate_mode is not None else 'left'
            self.vol_embedding = self.hps_ms.model.vol_embedding if self.hps_ms.model.vol_embedding is not None else False
            self.speech_encoder = self.hps_ms.model.speech_encoder if self.hps_ms.model.speech_encoder is not None else 'vec768l12'
 
        self.nsf_hifigan_enhance = nsf_hifigan_enhance
        if self.shallow_diffusion or self.only_diffusion:
            if os.path.exists(diffusion_model_path) and os.path.exists(diffusion_model_path):
                self.diffusion_model,self.vocoder,self.diffusion_args = load_model_vocoder(diffusion_model_path,self.dev,config_path=diffusion_config_path)
                if self.only_diffusion:
                    self.target_sample = self.diffusion_args.data.sampling_rate
                    self.hop_size = self.diffusion_args.data.block_size
                    self.spk2id = self.diffusion_args.spk
                    self.dtype = torch.float32
                    self.speech_encoder = self.diffusion_args.data.encoder
                    self.unit_interpolate_mode = self.diffusion_args.data.unit_interpolate_mode if self.diffusion_args.data.unit_interpolate_mode is not None else 'left'
                if spk_mix_enable:
                    self.diffusion_model.init_spkmix(len(self.spk2id))
            else:
                print("No diffusion model or config found. Shallow diffusion mode will False")
                self.shallow_diffusion = self.only_diffusion = False
                
        # load hubert and model
        if not self.only_diffusion:
            self.load_model(spk_mix_enable)
            self.hubert_model = utils.get_speech_encoder(self.speech_encoder,device=self.dev)
            self.volume_extractor = utils.Volume_Extractor(self.hop_size)
        else:
            self.hubert_model = utils.get_speech_encoder(self.diffusion_args.data.encoder,device=self.dev)
            self.volume_extractor = utils.Volume_Extractor(self.diffusion_args.data.block_size)
            
        if os.path.exists(cluster_model_path):
            if self.feature_retrieval:
                with open(cluster_model_path,"rb") as f:
                    self.cluster_model = pickle.load(f)
                self.big_npy = None
                self.now_spk_id = -1
            else:
                self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        else:
            self.feature_retrieval=False

        if self.shallow_diffusion :
            self.nsf_hifigan_enhance = False
        if self.nsf_hifigan_enhance:
            from modules.enhancer import Enhancer
            self.enhancer = Enhancer('nsf-hifigan', 'pretrain/nsf_hifigan/model',device=self.dev)
            
    def load_model(self, spk_mix_enable=False):
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)
        if spk_mix_enable:
            self.net_g_ms.EnableCharacterMix(len(self.spk2id), self.dev)

    def get_unit_f0(self, wav, tran, cluster_infer_ratio, speaker, f0_filter ,f0_predictor,cr_threshold=0.05):

        if not hasattr(self,"f0_predictor_object") or self.f0_predictor_object is None or f0_predictor != self.f0_predictor_object.name:
            self.f0_predictor_object = utils.get_f0_predictor(f0_predictor,hop_length=self.hop_size,sampling_rate=self.target_sample,device=self.dev,threshold=cr_threshold)
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)

        if f0_filter and sum(f0) == 0:
            raise F0FilterException("No voice detected")
        f0 = torch.FloatTensor(f0).to(self.dev)
        uv = torch.FloatTensor(uv).to(self.dev)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav = torch.from_numpy(wav).to(self.dev)
        if not hasattr(self,"audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample, 16000).to(self.dev)
        wav16k = self.audio16k_resample_transform(wav[None,:])[0]
        
        c = self.hubert_model.encoder(wav16k)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1],self.unit_interpolate_mode)

        if cluster_infer_ratio !=0:
            if self.feature_retrieval:
                speaker_id = self.spk2id.get(speaker)
                if not speaker_id and type(speaker) is int:
                    if len(self.spk2id.__dict__) >= speaker:
                        speaker_id = speaker
                if speaker_id is None:
                    raise RuntimeError("The name you entered is not in the speaker list!")
                feature_index = self.cluster_model[speaker_id]
                feat_np = np.ascontiguousarray(c.transpose(0,1).cpu().numpy())
                if self.big_npy is None or self.now_spk_id != speaker_id:
                   self.big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
                   self.now_spk_id = speaker_id
                print("starting feature retrieval...")
                score, ix = feature_index.search(feat_np, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                c = cluster_infer_ratio * npy + (1 - cluster_infer_ratio) * feat_np
                c = torch.FloatTensor(c).to(self.dev).transpose(0,1)
                print("end feature retrieval...")
            else:
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
              k_step = 100,
              frame = 0,
              spk_mix = False,
              second_encoding = False,
              loudness_envelope_adjustment = 1
              ):
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(raw_path)
        if not hasattr(self,"audio_resample_transform") or self.audio16k_resample_transform.orig_freq != sr:
            self.audio_resample_transform = torchaudio.transforms.Resample(sr,self.target_sample)
        wav = self.audio_resample_transform(wav).numpy()[0]
        if spk_mix:
            c, f0, uv = self.get_unit_f0(wav, tran, 0, None, f0_filter,f0_predictor,cr_threshold=cr_threshold)
            n_frames = f0.size(1)
            sid = speaker[:, frame:frame+n_frames].transpose(0,1)
        else:
            speaker_id = self.spk2id.get(speaker)
            if not speaker_id and type(speaker) is int:
                if len(self.spk2id.__dict__) >= speaker:
                    speaker_id = speaker
            if speaker_id is None:
                raise RuntimeError("The name you entered is not in the speaker list!")
            sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
            c, f0, uv = self.get_unit_f0(wav, tran, cluster_infer_ratio, speaker, f0_filter,f0_predictor,cr_threshold=cr_threshold)
            n_frames = f0.size(1)
        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)
        with torch.no_grad():
            start = time.time()
            vol = None
            if not self.only_diffusion:
                vol = self.volume_extractor.extract(torch.FloatTensor(wav).to(self.dev)[None,:])[None,:].to(self.dev) if self.vol_embedding else None
                audio,f0 = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noice_scale=noice_scale,vol=vol)
                audio = audio[0,0].data.float()
                audio_mel = self.vocoder.extract(audio[None,:],self.target_sample) if self.shallow_diffusion else None
            else:
                audio = torch.FloatTensor(wav).to(self.dev)
                audio_mel = None
            if self.dtype != torch.float32:
                c = c.to(torch.float32)
                f0 = f0.to(torch.float32)
                uv = uv.to(torch.float32)
            if self.only_diffusion or self.shallow_diffusion:
                vol = self.volume_extractor.extract(audio[None,:])[None,:,None].to(self.dev) if vol is None else vol[:,:,None]
                if self.shallow_diffusion and second_encoding:
                    if not hasattr(self,"audio16k_resample_transform"):
                        self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample, 16000).to(self.dev)
                    audio16k = self.audio16k_resample_transform(audio[None,:])[0]
                    c = self.hubert_model.encoder(audio16k)
                    c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[1],self.unit_interpolate_mode)
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
            if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance(
                                    audio[None,:], 
                                    self.target_sample, 
                                    f0[:,:,None], 
                                    self.hps_ms.data.hop_length, 
                                    adaptive_key = enhancer_adaptive_key)
            if loudness_envelope_adjustment != 1:
                audio = utils.change_rms(wav,self.target_sample,audio,self.target_sample,loudness_envelope_adjustment)
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1], n_frames

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
                        k_step = 100,
                        use_spk_mix = False,
                        second_encoding = False,
                        loudness_envelope_adjustment = 1
                        ):
        if use_spk_mix:
            if len(self.spk2id) == 1:
                spk = self.spk2id.keys()[0]
                use_spk_mix = False
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip_seconds*audio_sr)
        lg_size = int(lg_num*audio_sr)
        lg_size_r = int(lg_size*lgr_num)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0

        if use_spk_mix:
            assert len(self.spk2id) == len(spk)
            audio_length = 0
            for (slice_tag, data) in audio_data:
                aud_length = int(np.ceil(len(data) / audio_sr * self.target_sample))
                if slice_tag:
                    audio_length += aud_length // self.hop_size
                    continue
                if per_size != 0:
                    datas = split_list_by_n(data, per_size,lg_size)
                else:
                    datas = [data]
                for k,dat in enumerate(datas):
                    pad_len = int(audio_sr * pad_seconds)
                    per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample))
                    a_length = per_length + 2 * pad_len
                    audio_length += a_length // self.hop_size
            audio_length += len(audio_data)
            spk_mix_tensor = torch.zeros(size=(len(spk), audio_length)).to(self.dev)
            for i in range(len(spk)):
                last_end = None
                for mix in spk[i]:
                    if mix[3]<0. or mix[2]<0.:
                        raise RuntimeError("mix value must higer Than zero!")
                    begin = int(audio_length * mix[0])
                    end = int(audio_length * mix[1])
                    length = end - begin
                    if length<=0:                        
                        raise RuntimeError("begin Must lower Than end!")
                    step = (mix[3] - mix[2])/length
                    if last_end is not None:
                        if last_end != begin:
                            raise RuntimeError("[i]EndTime Must Equal [i+1]BeginTime!")
                    last_end = end
                    if step == 0.:
                        spk_mix_data = torch.zeros(length).to(self.dev) + mix[2]
                    else:
                        spk_mix_data = torch.arange(mix[2],mix[3],step).to(self.dev)
                    if(len(spk_mix_data)<length):
                        num_pad = length - len(spk_mix_data)
                        spk_mix_data = torch.nn.functional.pad(spk_mix_data, [0, num_pad], mode="reflect").to(self.dev)
                    spk_mix_tensor[i][begin:end] = spk_mix_data[:length]

            spk_mix_ten = torch.sum(spk_mix_tensor,dim=0).unsqueeze(0).to(self.dev)
            # spk_mix_tensor[0][spk_mix_ten<0.001] = 1.0
            for i, x in enumerate(spk_mix_ten[0]):
                if x == 0.0:
                    spk_mix_ten[0][i] = 1.0
                    spk_mix_tensor[:,i] = 1.0 / len(spk)
            spk_mix_tensor = spk_mix_tensor / spk_mix_ten
            if not ((torch.sum(spk_mix_tensor,dim=0) - 1.)<0.0001).all():
                raise RuntimeError("sum(spk_mix_tensor) not equal 1")
            spk = spk_mix_tensor

        global_frame = 0
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size,lg_size)
            else:
                datas = [data]
            for k,dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if clip_seconds!=0 else length
                if clip_seconds!=0: 
                    print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                # padd
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr, out_frame = self.infer(spk, tran, raw_path,
                                                    cluster_infer_ratio=cluster_infer_ratio,
                                                    auto_predict_f0=auto_predict_f0,
                                                    noice_scale=noice_scale,
                                                    f0_predictor = f0_predictor,
                                                    enhancer_adaptive_key = enhancer_adaptive_key,
                                                    cr_threshold = cr_threshold,
                                                    k_step = k_step,
                                                    frame = global_frame,
                                                    spk_mix = use_spk_mix,
                                                    second_encoding = second_encoding,
                                                    loudness_envelope_adjustment = loudness_envelope_adjustment
                                                    )
                global_frame += out_frame
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
            
