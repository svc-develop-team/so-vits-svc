import argparse
import time
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime as ort
import onnxoptimizer
import torch
from model_onnx_48k import SynthesizerTrn
import utils
from hubert import hubert_model_onnx

def main(HubertExport,NetExport):

    path = "NyaruTaffy"

    if(HubertExport):
        device = torch.device("cuda")
        hubert_soft = hubert_model_onnx.hubert_soft("hubert/model.pt")
        test_input = torch.rand(1, 1, 16000)
        input_names = ["source"]
        output_names = ["embed"]
        torch.onnx.export(hubert_soft.to(device),
                        test_input.to(device),
                        "hubert3.0.onnx",
                        dynamic_axes={
                            "source": {
                                2: "sample_length"
                            }
                        },
                        verbose=False,
                        opset_version=13,
                        input_names=input_names,
                        output_names=output_names)
    if(NetExport):
        device = torch.device("cuda")
        hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        SVCVITS = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        _ = utils.load_checkpoint(f"checkpoints/{path}/model.pth", SVCVITS, None)
        _ = SVCVITS.eval().to(device)
        for i in SVCVITS.parameters():
            i.requires_grad = False
        test_hidden_unit = torch.rand(1, 50, 256)
        test_lengths = torch.LongTensor([50])
        test_pitch = torch.rand(1, 50)
        test_sid = torch.LongTensor([0])
        input_names = ["hidden_unit", "lengths", "pitch", "sid"]
        output_names = ["audio", ]
        SVCVITS.eval()
        torch.onnx.export(SVCVITS,
                        (
                            test_hidden_unit.to(device),
                            test_lengths.to(device),
                            test_pitch.to(device),
                            test_sid.to(device)
                        ),
                        f"checkpoints/{path}/model.onnx",
                        dynamic_axes={
                            "hidden_unit": [0, 1],
                            "pitch": [1]
                        },
                        do_constant_folding=False,
                        opset_version=16,
                        verbose=False,
                        input_names=input_names,
                        output_names=output_names)


if __name__ == '__main__':
    main(False,True)
