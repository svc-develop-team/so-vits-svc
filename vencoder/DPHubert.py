import torch

from vencoder.dphubert.model import wav2vec2_model
from vencoder.encoder import SpeechEncoder


class DPHubert(SpeechEncoder):
    def __init__(self, vec_path="pretrain/DPHuBERT-sp0.75.pth", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        ckpt = torch.load(vec_path)
        self.hidden_dim = 768
        self.model = wav2vec2_model(**ckpt["config"]).to(self.dev)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None, :]
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model(feats)[0]
                return units.transpose(1,2)
