from vencoder.encoder import SpeechEncoder
import torch

class ContentVec768L12(SpeechEncoder):
    def __init__(self,vec_path = "pretrain/checkpoint_best_legacy_500.pt"):
        print("load model(s) from {}".format(vec_path))
        from fairseq import checkpoint_utils
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        self.hidden_dim = 768
        self.model = models[0]
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
          feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
          logits = self.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)