import os
import argparse
import re

from tqdm import tqdm
from random import shuffle
import json
import wave

config_template = json.load(open("configs_template/config_template.json"))

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # get audio frames
        n_frames = wav_file.getnframes()
        # get sampling rate
        framerate = wav_file.getframerate()
        # calculate duration in seconds
        duration = n_frames / float(framerate)
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
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
                print(f"Warning: The file name of {file} contains non-alphanumeric and underscores, which may cause issues. (or maybe not)")
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

    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
	
    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
