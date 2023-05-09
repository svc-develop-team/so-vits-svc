import io
import logging
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # Required
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_0.pth",
                        help='Path to the model.')
    parser.add_argument('-c', '--config_path', type=str, default="configs/config.json",
                        help='Path to the configuration file.')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['nen'],
                        help='Target speaker name for conversion.')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"],
                        help='A list of wav file names located in the raw folder.')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0],
                        help='Pitch adjustment, supports positive and negative (semitone) values.')

    # Optional
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='Automatic pitch prediction for voice conversion. Do not enable this when converting songs as it can cause serious pitch issues.')
    parser.add_argument('-cl', '--clip', type=float, default=0,
                        help='Voice forced slicing. Set to 0 to turn off(default), duration in seconds.')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0,
                        help='The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use. Default 0.')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt",
                        help='Path to the clustering model. Fill in any value if clustering is not trained.')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0,
                        help='Proportion of the clustering solution, range 0-1. Fill in 0 if the clustering model is not trained.')
    parser.add_argument('-fmp', '--f0_mean_pooling', action='store_true', default=False,
                        help='Apply mean filter (pooling) to f0, which may improve some hoarse sounds. Enabling this option will reduce inference speed.')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False,
                        help='Whether to use NSF_HIFIGAN enhancer. This option has certain effect on sound quality enhancement for some models with few training sets, but has negative effect on well-trained models, so it is turned off by default.')

    # generally keep default
    parser.add_argument('-sd', '--slice_db', type=int, default=-40,
                        help='Loudness for automatic slicing. For noisy audio it can be set to -30')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Device used for inference. None means auto selecting.')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4,
                        help='Affect pronunciation and sound quality.')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5,
                        help='Due to unknown reasons, there may be abnormal noise at the beginning and end. It will disappear after padding a short silent segment.')
    parser.add_argument('-wf', '--wav_format', type=str, default='flac',
                        help='output format')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75,
                        help='Proportion of cross length retention, range (0-1]. After forced slicing, the beginning and end of each segment need to be discarded.')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0,
                        help='Adapt the enhancer to a higher range of sound. The unit is the semitones, default 0.')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,
                        help='F0 Filtering threshold: This parameter is valid only when f0_mean_pooling is enabled. Values range from 0 to 1. Reducing this value reduces the probability of being out of tune, but increases matte.')


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    F0_mean_pooling = args.f0_mean_pooling
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold

    svc_model = Svc(args.model_path, args.config_path, args.device, args.cluster_model_path,enhance)
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
        per_size = int(clip*audio_sr)
        lg_size = int(lg*audio_sr)
        lg_size_r = int(lg_size*lgr)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg_2 = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0

        for spk in spk_list:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
                
                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                    audio.extend(list(infer_tool.pad_array(_audio, length)))
                    continue
                if per_size != 0:
                    datas = infer_tool.split_list_by_n(data, per_size,lg_size)
                else:
                    datas = [data]
                for k,dat in enumerate(datas):
                    per_length = int(np.ceil(len(dat) / audio_sr * svc_model.target_sample)) if clip!=0 else length
                    if clip!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, dat, audio_sr, format="wav")
                    raw_path.seek(0)
                    out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                        cluster_infer_ratio=cluster_infer_ratio,
                                                        auto_predict_f0=auto_predict_f0,
                                                        noice_scale=noice_scale,
                                                        F0_mean_pooling = F0_mean_pooling,
                                                        enhancer_adaptive_key = enhancer_adaptive_key,
                                                        cr_threshold = cr_threshold
                                                        )
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]
                    _audio = infer_tool.pad_array(_audio, per_length)
                    if lg_size!=0 and k!=0:
                        lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr != 1 else audio[-lg_size:]
                        lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr != 1 else _audio[0:lg_size]
                        lg_pre = lg1*(1-lg_2)+lg2*lg_2
                        audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr != 1 else audio[0:-lg_size]
                        audio.extend(lg_pre)
                        _audio = _audio[lg_size_c_l+lg_size_r:] if lgr != 1 else _audio[lg_size:]
                    audio.extend(list(_audio))
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            res_path = f'./results/{clean_name}_{key}_{spk}{cluster_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
