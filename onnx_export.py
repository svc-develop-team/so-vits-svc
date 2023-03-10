import torch
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from fairseq import checkpoint_utils
from onnxexport.model_onnx import SynthesizerTrn
import utils

def get_hubert_model():
    vec_path = "hubert/checkpoint_best_legacy_500.pt"
    print("load model(s) from {}".format(vec_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path],
        suffix="",
    )
    model = models[0]
    model.eval()
    return model


def main(HubertExport, NetExport):
    path = "SoVits4.0"

    '''if HubertExport:
        device = torch.device("cpu")
        vec_path = "hubert/checkpoint_best_legacy_500.pt"
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        original = models[0]
        original.eval()
        model = original
        test_input = torch.rand(1, 1, 16000)
        model(test_input)
        torch.onnx.export(model,
                          test_input,
                          "hubert4.0.onnx",
                          export_params=True,
                          opset_version=16,
                          do_constant_folding=True,
                          input_names=['source'],
                          output_names=['embed'],
                          dynamic_axes={
                              'source':
                                  {
                                      2: "sample_length"
                                  },
                          }
                          )'''
    if NetExport:
        device = torch.device("cpu")
        hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        SVCVITS = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", SVCVITS, None)
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False
        test_hidden_unit = torch.rand(1, 10, 256)
        test_pitch = torch.rand(1, 10)
        test_mel2ph = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
        test_uv = torch.ones(1, 10, dtype=torch.float32)
        test_noise = torch.randn(1, 192, 10)
        test_sid = torch.LongTensor([0])
        input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
        output_names = ["audio", ]
        SVCVITS.eval()
        torch.onnx.export(SVCVITS,
                          (
                              test_hidden_unit.to(device),
                              test_pitch.to(device),
                              test_mel2ph.to(device),
                              test_uv.to(device),
                              test_noise.to(device),
                              test_sid.to(device)
                          ),
                          f"checkpoints/{path}/model.onnx",
                          dynamic_axes={
                              "c": [0, 1],
                              "f0": [1],
                              "mel2ph": [1],
                              "uv": [1],
                              "noise": [2],
                          },
                          do_constant_folding=False,
                          opset_version=16,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names)


if __name__ == '__main__':
    main(False, True)
