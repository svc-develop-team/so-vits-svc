import sys
from functools import reduce

import librosa
import numpy as np
import torch
from torch.nn.modules.module import _addindent

from .constants import *  # noqa: F403


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

    
def to_local_average_cents(salience, center=None, thred=0.05):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                20 * torch.arange(N_CLASS) + CONST).to(salience.device)  # noqa: F405

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = torch.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = torch.sum(salience)
        return product_sum / weight_sum if torch.max(salience) > thred else 0
    if salience.ndim == 2:
        return torch.Tensor([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])]).to(salience.device)

    raise Exception("label should be either 1d or 2d ndarray")

def to_viterbi_cents(salience, thred=0.05):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = torch.meshgrid(range(N_CLASS), range(N_CLASS))  # noqa: F405
        transition = torch.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    # Convert to probability
    prob = salience.T
    prob = prob / prob.sum(axis=0)    

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob.detach().cpu().numpy(), to_viterbi_cents.transition).astype(np.int64)

    return torch.Tensor([to_local_average_cents(salience[i, :], path[i], thred) for i in
                     range(len(path))]).to(salience.device)
                     