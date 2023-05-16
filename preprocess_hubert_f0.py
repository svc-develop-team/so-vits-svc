import math
import multiprocessing
import os
import argparse
from random import shuffle
import random

import torch
from glob import glob
from tqdm import tqdm
from modules.mel_processing import spectrogram_torch
import json

import utils
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import diffusion.logger.utils as du 
from diffusion.vocoder import Vocoder

import librosa
import numpy as np

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, hmodel,f0p,diff=False,mel_extractor=None):
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)
        
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))
    
    
    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )
        
        #audio_norm = audio / hps.data.max_wav_value
        
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


def process_batch(filenames,f0p,diff=False,mel_extractor=None):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_speech_encoder(speech_encoder,device=device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, hmodel,f0p,diff,mel_extractor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument( 
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument( 
        '--f0_predictor', type=str, default="dio", help='Select F0 predictor, can select crepe,pm,dio,harvest, default pm(note: crepe is original F0 using mean filter)'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    f0p = args.f0_predictor
    print(speech_encoder)
    print(f0p)
    if args.use_diff:
        print("use_diff")
        print("Loading Mel Extractor...")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device = device)
        print("Loaded Mel Extractor.")
    else:
        mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)
    
    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,f0p,args.use_diff,mel_extractor)) for chunk in chunks
    ]
    for p in processes:
        p.start()
