import argparse
import json
import os
import re
import wave
from random import shuffle

from tqdm import tqdm

import diffusion.logger.utils as du

config_template = json.load(open("configs_template/config_template.json"))

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # 获取音频帧数
        n_frames = wav_file.getnframes()
        # 获取采样率
        framerate = wav_file.getframerate()
        # 计算时长（秒）
        duration = n_frames / float(framerate)
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    args = parser.parse_args()
    
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = ["/".join([args.source_dir, speaker, i]) for i in os.listdir(os.path.join(args.source_dir, speaker))]
        new_wavs = []
        for file in wavs:
            if not file.endswith("wav"):
                continue
            if not pattern.match(file):
                print(f"warning：文件名{file}中包含非字母数字下划线，可能会导致错误。（也可能不会）")
            if get_wav_duration(file) < 0.3:
                print("skip too short audio:", file)
                continue
            new_wavs.append(file)
        wavs = new_wavs
        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)
            
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")
        
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")


    d_config_template = du.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id
    d_config_template["data"]["encoder"] = args.speech_encoder
    d_config_template["spk"] = spk_dict
    
    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
    config_template["model"]["speech_encoder"] = args.speech_encoder
    
    if args.speech_encoder == "vec768l12" or args.speech_encoder == "dphubert" or args.speech_encoder == "wavlmbase+":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 768
        d_config_template["data"]["encoder_out_channels"] = 768
    elif args.speech_encoder == "vec256l9" or args.speech_encoder == 'hubertsoft':
        config_template["model"]["ssl_dim"] = config_template["model"]["gin_channels"] = 256
        d_config_template["data"]["encoder_out_channels"] = 256
    elif args.speech_encoder == "whisper-ppg" or args.speech_encoder == 'cnhubertlarge':
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1024
        d_config_template["data"]["encoder_out_channels"] = 1024
    elif args.speech_encoder == "whisper-ppg-large":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1280
        d_config_template["data"]["encoder_out_channels"] = 1280
        
    if args.vol_aug:
        config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True

    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    print("Writing configs/diffusion.yaml")
    du.save_config("configs/diffusion.yaml",d_config_template)
