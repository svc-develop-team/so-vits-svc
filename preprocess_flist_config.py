import argparse
import json
import os
import re
import wave
from random import shuffle

import logger
from tqdm import tqdm

import diffusion.logger.utils as du

import time 

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    args = parser.parse_args()
    
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    with logger.Progress() as progress:
        for speaker in progress.track(os.listdir(args.source_dir), description="Processing Speakers"):
            spk_dict[speaker] = spk_id
            spk_id += 1
            wavs = []

            for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
                if not file_name.endswith("wav"):
                    continue
                if file_name.startswith("."):
                    continue

                file_path = "/".join([args.source_dir, speaker, file_name])

                if not pattern.match(file_name):
                    logger.warning("Detected non-ASCII file name: " + file_path)

                if get_wav_duration(file_path) < 0.3:
                    logger.info("Skip too short audio: " + file_path)
                    continue

                wavs.append(file_path)

            shuffle(wavs)
            train += wavs[2:]
            val += wavs[:2]

        shuffle(train)
        shuffle(val)

        logger.info("Writing " + args.train_list)
        with open(args.train_list, "w") as f:
            for fname in progress.track(train, description="Writing train list"):
                wavpath = fname
                f.write(wavpath + "\n")

        logger.info("Writing " + args.val_list)
        with open(args.val_list, "w") as f:
            for fname in progress.track(val, description="Writing val list"):
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

        if args.tiny:
            config_template["model"]["filter_channels"] = 512

        logger.info("Writing to configs/config.json")
        with open("configs/config.json", "w") as f:
            json.dump(config_template, f, indent=2)
        logger.info("Writing to configs/diffusion.yaml")
        du.save_config("configs/diffusion.yaml",d_config_template)
