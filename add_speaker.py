import os
import argparse
from tqdm import tqdm
from random import shuffle
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="./filelists/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=str, default="./dataset/32k", help="path to source dir")
    args = parser.parse_args()

    previous_config = json.load(open("configs/config.json", "rb"))

    train = []
    val = []
    test = []
    idx = 0
    spk_dict = previous_config["spk"]
    spk_id = max([i for i in spk_dict.values()]) + 1
    for speaker in tqdm(os.listdir(args.source_dir)):
        if speaker not in spk_dict.keys():
            spk_dict[speaker] = spk_id
            spk_id += 1
        wavs = [os.path.join(args.source_dir, speaker, i)for i in os.listdir(os.path.join(args.source_dir, speaker))]
        wavs = [i for i in wavs if i.endswith("wav")]
        shuffle(wavs)
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]

    assert previous_config["model"]["n_speakers"] > len(spk_dict.keys())
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

    previous_config["spk"] = spk_dict

    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(previous_config, f, indent=2)
