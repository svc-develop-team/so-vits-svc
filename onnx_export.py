import torch
from onnxexport.model_onnx import SynthesizerTrn
import utils

def main(NetExport):
    path = "SoVits4.0V2"
    if NetExport:
        device = torch.device("cpu")
        hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        SVCVITS = SynthesizerTrn(
            hps)
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", SVCVITS, None)
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False
        
        test_hidden_unit = torch.rand(1, 10, 256) # rand
        test_pitch = torch.rand(1, 10) # rand
        test_mel2ph = torch.arange(0, 10, dtype=torch.int64)[None]
        #test_mel2ph = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).unsqueeze(0)
        test_uv = torch.zeros(1, 2048, 10, dtype=torch.float32)
        test_uv += 0.0064
        test_noise = torch.randn(1, 192, 10) # randn
        test_sid = torch.LongTensor([0])
        input_names = ["c", "f0", "mel2ph", "t_window", "noise", "sid"]
        output_names = ["audio", ]
        
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
                              "t_window": [2],
                              "noise": [2],
                          },
                          do_constant_folding=False,
                          opset_version=16,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names)


if __name__ == '__main__':
    main(True)
