from collections import OrderedDict

import torch

import utils
from models import SynthesizerTrn


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ','.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    hps = utils.get_hparams_from_file(config)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model)

    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hps.train.learning_rate,
                                betas=hps.train.betas,
                                eps=hps.train.eps)

    state_dict_g = torch.load(input_model, map_location="cpu")
    new_dict_g = copyStateDict(state_dict_g)
    keys = []
    for k, v in new_dict_g['model'].items():
        if "enc_q" in k: continue  # noqa: E701
        keys.append(k)
    
    new_dict_g = {k: new_dict_g['model'][k].half() for k in keys} if ishalf else {k: new_dict_g['model'][k] for k in keys}

    torch.save(
        {
            'model': new_dict_g,
            'iteration': 0,
            'optimizer': optim_g.state_dict(),
            'learning_rate': 0.0001
        }, output_model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='configs/config.json')
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument('-hf', '--half', action='store_true', default=False, help='Save as FP16')
    
    args = parser.parse_args()

    output = args.output

    if output is None:
        import os.path
        filename, ext = os.path.splitext(args.input)
        half = "_half" if args.half else ""
        output = filename + "_release" + half + ext

    removeOptimizer(args.config, args.input, args.half, output)