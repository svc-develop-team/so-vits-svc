import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import *  # noqa: F403
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_cents, to_viterbi_cents


class RMVPE:
    def __init__(self, model_path, device=None, dtype = torch.float32, hop_length=160):
        self.resample_kernel = {}
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(ckpt['model'])
        model = model.to(dtype).to(self.device)
        model.eval()
        self.model = model
        self.dtype = dtype
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='reflect')
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            cents_pred = to_viterbi_cents(hidden, thred=thred)
        else:
            cents_pred = to_local_average_cents(hidden, thred=thred)
        f0 = torch.Tensor([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred]).to(self.device)
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        audio = audio.unsqueeze(0).to(self.dtype).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel_extractor = self.mel_extractor.to(self.device)
        mel = mel_extractor(audio_res, center=True).to(self.dtype)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden.squeeze(0), thred=thred, use_viterbi=use_viterbi)
        return f0
