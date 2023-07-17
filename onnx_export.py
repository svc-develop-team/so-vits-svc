import json

import torch

import utils
from onnxexport.model_onnx_speaker_mix import SynthesizerTrn


def main():
    path = "crs"

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
    
    num_frames = 200

    test_hidden_unit = torch.rand(1, num_frames, SVCVITS.gin_channels)
    test_pitch = torch.rand(1, num_frames)
    test_vol = torch.rand(1, num_frames)
    test_mel2ph = torch.LongTensor(torch.arange(0, num_frames)).unsqueeze(0)
    test_uv = torch.ones(1, num_frames, dtype=torch.float32)
    test_noise = torch.randn(1, 192, num_frames)
    test_sid = torch.LongTensor([0])
    export_mix = True
    if len(hps.spk) < 2:
        export_mix = False
    
    if export_mix:
        spk_mix = []
        n_spk = len(hps.spk)
        for i in range(n_spk):
            spk_mix.append(1.0/float(n_spk))
        test_sid = torch.tensor(spk_mix)
        SVCVITS.export_chara_mix(hps.spk)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(num_frames, 1)
    
    SVCVITS.eval()

    if export_mix:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2],
            "sid":[0]
        }
    else:
        daxes = {
            "c": [0, 1],
            "f0": [1],
            "mel2ph": [1],
            "uv": [1],
            "noise": [2]
        }
    
    input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
    output_names = ["audio", ]

    if SVCVITS.vol_embedding:
        input_names.append("vol")
        vol_dadict = {"vol" : [1]}
        daxes.update(vol_dadict)
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device)
        )
    else:
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device)
        )

    # SVCVITS = torch.jit.script(SVCVITS)
    SVCVITS(test_hidden_unit.to(device),
            test_pitch.to(device),
            test_mel2ph.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device))

    SVCVITS.dec.OnnxExport()

    torch.onnx.export(
        SVCVITS,
        test_inputs,
        f"checkpoints/{path}/{path}_SoVits.onnx",
        dynamic_axes=daxes,
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names
    )

    vec_lay = "layer-12" if SVCVITS.gin_channels == 768 else "layer-9"
    spklist = []
    for key in hps.spk.keys():
        spklist.append(key)

    MoeVSConf = {
        "Folder" : f"{path}",
        "Name" : f"{path}",
        "Type" : "SoVits",
        "Rate" : hps.data.sampling_rate,
        "Hop" : hps.data.hop_length,
        "Hubert": f"vec-{SVCVITS.gin_channels}-{vec_lay}",
        "SoVits4": True,
        "SoVits3": False,
        "CharaMix": export_mix,
        "Volume": SVCVITS.vol_embedding,
        "HiddenSize": SVCVITS.gin_channels,
        "Characters": spklist
    }

    with open(f"checkpoints/{path}.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)


if __name__ == '__main__':
    main()
