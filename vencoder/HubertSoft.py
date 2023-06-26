import torch

from vencoder.encoder import SpeechEncoder
from vencoder.hubert import hubert_model


class HubertSoft(SpeechEncoder):
    def __init__(self, vec_path="pretrain/hubert-soft-0d54a1f4.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        hubert_soft = hubert_model.hubert_soft(vec_path)
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = 256
        self.model = hubert_soft.to(self.dev)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None,None,:]  
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model.units(feats)
                return units.transpose(1,2)
