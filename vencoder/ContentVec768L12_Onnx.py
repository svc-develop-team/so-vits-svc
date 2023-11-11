import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class ContentVec768L12_Onnx(SpeechEncoder):
    def __init__(self, vec_path="pretrain/vec-768-layer-12.onnx", device=None, log=True):
        super().__init__()
        import logger
        if log:
            logger.info("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
