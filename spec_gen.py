from data_utils import TextAudioSpeakerLoader
import json
from tqdm import tqdm

from utils import HParams

config_path = 'configs/config.json'
with open(config_path, "r") as f:
    data = f.read()
config = json.loads(data)
hps = HParams(**config)

train_dataset = TextAudioSpeakerLoader("filelists/train.txt", hps)
eval_dataset = TextAudioSpeakerLoader("filelists/val.txt", hps)

for _ in tqdm(train_dataset):
    pass
for _ in tqdm(eval_dataset):
    pass
