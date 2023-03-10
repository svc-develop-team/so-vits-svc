import os
import librosa
import pyworld
import utils
import numpy as np
from scipy.io import wavfile


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path):
        x, sr = librosa.load(path, sr=self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * self.hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return f0

    # for numpy # code from diffsinger
    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    # for tensor # code from diffsinger
    def coarse_f0_ts(self, f0):
        f0_mel = 1127 * (1 + f0 / 700).log()
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = (f0_mel + 0.5).long()
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, self.fs, wav.astype(np.int16))


if __name__ == "__main__":
    wavPath = "./data/waves"
    outPath = "./data/label"
    if not os.path.exists("./data/label"):
        os.mkdir("./data/label")

    # define model and load checkpoint
    hps = utils.get_hparams_from_file("./configs/singing_base.json")
    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)
    vits_file = open("./filelists/vc_file.txt", "w", encoding="utf-8")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    audio_path = f"./{wavPath}/{spks}/{file}.wav"
                    featur_pit = featureInput.compute_f0(audio_path)
                    coarse_pit = featureInput.coarse_f0(featur_pit)
                    np.save(
                        f"{outPath}/{spks}/{file}_pitch.npy",
                        coarse_pit,
                        allow_pickle=False,
                    )
                    np.save(
                        f"{outPath}/{spks}/{file}_nsff0.npy",
                        featur_pit,
                        allow_pickle=False,
                    )

                    path_audio = f"./data/waves/{spks}/{file}.wav"
                    path_spkid = f"./data/spkid/{spks}.npy"
                    path_label = (
                        f"./data/phone/{spks}/{file}.npy"  # phone means ppg & hubert
                    )
                    path_pitch = f"./data/label/{spks}/{file}_pitch.npy"
                    path_nsff0 = f"./data/label/{spks}/{file}_nsff0.npy"
                    print(
                        f"{path_audio}|{path_spkid}|{path_label}|{path_pitch}|{path_nsff0}",
                        file=vits_file,
                    )

    vits_file.close()
