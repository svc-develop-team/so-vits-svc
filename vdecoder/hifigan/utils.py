import glob
import os

# matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch
from torch.nn.utils import weight_norm


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def del_old_checkpoints(cp_dir, prefix, n_models=2):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern) # get checkpoint paths
    cp_list = sorted(cp_list)# sort by iter
    if len(cp_list) > n_models: # if more than n_models models are found
        for cp in cp_list[:-n_models]:# delete the oldest models other than lastest n_models
            open(cp, 'w').close()# empty file contents
            os.unlink(cp)# delete file (move to trash when using Colab)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

