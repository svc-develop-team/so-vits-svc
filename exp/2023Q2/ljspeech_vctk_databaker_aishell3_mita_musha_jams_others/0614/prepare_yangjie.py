import json
import os
import shutil
from collections import defaultdict

import re
import requests
import textgrid
import torchaudio
import librosa
import numpy as np
from scipy.io import wavfile

input_textgrid_fpath = '/home/tmp-yichao.hu/workspace/TTS-datasets/projects/2023Q2/0420/yangjie/step4/3.TextGrid'
input_wav_fpath = '/home/tmp-yichao.hu/workspace/TTS-datasets/projects/2023Q2/0420/yangjie/step5/3_rx9_enhancedV1.wav'

# input_textgrid_fpath = '/home/tmp-yichao.hu/workspace/TTS-datasets/projects/2023Q2/0420/yangjie/step4/1.TextGrid'
# input_wav_fpath = '/home/tmp-yichao.hu/workspace/TTS-datasets/projects/2023Q2/0420/yangjie/step5/1_rx9_enhancedV1.wav'

output_dir = '/home/tmp-yi.liu/yangjie/step5'

basename = os.path.basename(input_textgrid_fpath).rsplit(".", 1)[0]
tg = textgrid.TextGrid.fromFile(input_textgrid_fpath)
assert len(tg) == 6, f"Wrong label {input_textgrid_fpath}"
index = 0
tg1_intervals = list(tg[0])
tg2_intervals = list(tg[1])
tg3_intervals = list(tg[2])
leaf_text_intervals = list(tg[3])
leaf_pinyin_intervals = list(tg[4])
leaf_mos_intervals = list(tg[5])

class UtteranceNode:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
        self.children = []
        self.pinyin_seq = None
        self.mos = 5

    def update_text_mos_pinyin(self, node):
        assert isinstance(node, UtteranceNode)
        if self.text != node.text:
            print("update [{}] to [{}]".format(self.text, node.text))
        self.text = node.text
        self.pinyin_seq = node.pinyin_seq
        self.mos = node.mos

    def add_child(self, node):
        assert isinstance(node, UtteranceNode)
        self.children.append(node)

    def validate(self):
        cur_text = self.text
        if not self.children:
            return
        children_text = "".join([c.text for c in self.children])
        assert children_text == cur_text, "{} != {}".format(children_text, cur_text)
        for child in self.children:
            assert self.start <= child.start and self.end >= child.end
            child.validate()

    def update_from_leaves(self):
        cur_text = self.text
        if not self.children:
            return
        for c in self.children:
            if c.children:
                c.update_from_leaves()

        children_text = "".join(c.text for c in self.children)
        if children_text != cur_text:
            print("update from CHILDREN: [{}] -> [{}].".format(cur_text, children_text))
        self.text = children_text
        children_pinyin_seq = []
        children_mos = []
        for c in self.children:
            children_pinyin_seq.extend(c.pinyin_seq)
            children_mos.append(c.mos)
        self.pinyin_seq = children_pinyin_seq
        self.mos = min(children_mos)

    def collect_leaves(self):
        leaves = []
        if not self.children:
             leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.collect_leaves())
        return leaves


leaf_nodes = dict()
for text_interval in leaf_text_intervals:
    text = text_interval.mark.strip()
    if len(text) == 0:
        continue
    start_time = text_interval.minTime
    end_time = text_interval.maxTime

    node = UtteranceNode(text, start=start_time, end=end_time)

    while leaf_pinyin_intervals and len(leaf_pinyin_intervals[0].mark.strip()) == 0:
        leaf_pinyin_intervals.pop(0)
    pinyin_interval = leaf_pinyin_intervals.pop(0)
    assert pinyin_interval.minTime == node.start
    assert pinyin_interval.maxTime == node.end
    while leaf_mos_intervals and len(leaf_mos_intervals[0].mark.strip()) == 0:
        leaf_mos_intervals.pop(0)
    mos_interval = leaf_mos_intervals.pop(0)
    assert mos_interval.minTime == node.start
    assert mos_interval.maxTime == node.end
    node.pinyin_seq = pinyin_interval.mark.strip().split(" ")
    node.mos = int(mos_interval.mark.strip())

    leaf_nodes[(node.start, node.end)] = node


utterance_roots = []
for interval1 in tg1_intervals:
    text = interval1.mark.strip()
    if len(text) == 0:
        continue

    start_time = interval1.minTime
    end_time = interval1.maxTime

    utt_root = UtteranceNode(text, start=start_time, end=end_time)
    utterance_roots.append(utt_root)

    while tg2_intervals and tg2_intervals[0].maxTime <= end_time:
        interval2 = tg2_intervals.pop(0)
        interval2_text = interval2.mark.strip()
        if len(interval2_text) > 0:
            interval2_start_time = interval2.minTime
            interval2_end_time = interval2.maxTime
            utt_node_level2 = UtteranceNode(interval2_text, start=interval2_start_time, end=interval2_end_time)
            utt_root.add_child(utt_node_level2)

            while tg3_intervals and tg3_intervals[0].maxTime <= interval2_end_time:
                interval3 = tg3_intervals.pop(0)
                interval3_text = interval3.mark.strip()
                if len(interval3_text) > 0:
                    interval3_start_time = interval3.minTime
                    interval3_end_time = interval3.maxTime
                    utt_node_level3 = UtteranceNode(interval3_text, start=interval3_start_time, end=interval3_end_time)
                    utt_node_level2.add_child(utt_node_level3)

for utt_root in utterance_roots:
    utt_root.validate()
    for utt_leaf in utt_root.collect_leaves():
        assert (utt_leaf.start, utt_leaf.end) in leaf_nodes
        utt_leaf.update_text_mos_pinyin(leaf_nodes[(utt_leaf.start, utt_leaf.end)])
    utt_root.update_from_leaves()

wav_dir = os.path.join(output_dir, "LYG0002")
os.makedirs(wav_dir, exist_ok=True)
audio, sampling_rate = torchaudio.load(input_wav_fpath, normalize=False)
audio = audio[0]

def preprocess_and_normalize_audio(audio_file, to_wav_file, sample_rate=24000, max_wav_value=32768.0):

    audio_data, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

    volume = 0.8 * max_wav_value
    # load mp3 audio file to mono

    max_value = np.mean(np.sort(np.abs(audio_data))[-10:])
    max_value = max(max_value, 1e-8)
    norm_audio = audio_data * volume / max_value
    norm_audio = np.clip(norm_audio, -volume, volume)
    wavfile.write(to_wav_file, sample_rate, norm_audio.astype(np.int16))
    print("write file to {} with sample_rate {}".format(to_wav_file, sample_rate))


def dump_audio(uttid, start, end):
    start_sample = max(int(start * sampling_rate), 0)
    end_sample = min(int(end * sampling_rate), audio.shape[0])
    segment = audio[start_sample:end_sample].unsqueeze(0)
    temp_audio_fpath = "/tmp/temp.wav"
    torchaudio.save(temp_audio_fpath, segment, sample_rate=sampling_rate)
    preprocess_and_normalize_audio(temp_audio_fpath, os.path.join(wav_dir, "{}.wav".format(uttid)), sample_rate=sampling_rate)


def preprocess_utt_node(node: UtteranceNode, level=1):
    utt_id = "LYG0002{}{}_{}_{}".format(basename, str(node.start).replace(".", "_"), str(node.end).replace(".", "_"), level)
    print(utt_id, " ".join(node.pinyin_seq), node.text)
    dump_audio(utt_id, node.start, node.end)
    with open(os.path.join(wav_dir, "{}.lab".format(utt_id)), "w") as lab_f:
        lab_f.write(" ".join(node.pinyin_seq))


for utt_root in utterance_roots:
    preprocess_utt_node(utt_root, level=1)
    for level2_node in utt_root.children:
        preprocess_utt_node(level2_node, level=2)
        for level3_node in level2_node.children:
            preprocess_utt_node(level3_node, level=3)
