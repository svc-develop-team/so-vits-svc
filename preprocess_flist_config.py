import os
import argparse
import re

from tqdm import tqdm
from random import shuffle
import json
config_template = {
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 10000,
    "learning_rate": 1e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 12,
    "fp16_run": False,
    "lr_decay": 0.999875,
    "segment_size": 17920,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "use_sr": True,
    "max_speclen": 384,
    "port": "8001",
    "keep_ckpts": 5,  # Number of ckpts to keep, if 0, keep all
  },
  "data": {
    "training_files":"filelists/train.txt",
    "validation_files":"filelists/val.txt",
    "max_wav_value": 32768.0,
    "sampling_rate": 32000,
    "filter_length": 1280,
    "hop_length": 320,
    "win_length": 1280,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": None
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [10,8,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "gin_channels": 256,
    "ssl_dim": 256,
    "n_speakers": 0,
  },
  "spk":{
    "nen": 0,
    "paimon": 1,
    "yunhao": 2
  }
}

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="./filelists/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=str, default="./dataset/32k", help="path to source dir")
    args = parser.parse_args()
    
    train = []
    val = []
    test = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = ["/".join([args.source_dir, speaker, i]) for i in os.listdir(os.path.join(args.source_dir, speaker))]
        for wavpath in wavs:
            if not pattern.match(wavpath):
                print(f"warning：文件名{wavpath}中包含非字母数字下划线，可能会导致错误。（也可能不会）")
        if len(wavs) < 10:
            print(f"warning：{speaker}数据集数量小于10条，请补充数据")
        wavs = [i for i in wavs if i.endswith("wav")]
        shuffle(wavs)
        train += wavs[2:-2]
        val += wavs[:2]
        test += wavs[-2:]
    n_speakers = len(spk_dict.keys())*2
    shuffle(train)
    shuffle(val)
    shuffle(test)
            
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
            
    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            wavpath = fname
            f.write(wavpath + "\n")

    config_template["model"]["n_speakers"] = n_speakers
    config_template["spk"] = spk_dict
    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
