import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from multiprocessing import cpu_count

import faiss
import librosa
import numpy as np
import torch
from scipy.io.wavfile import read
from sklearn.cluster import MiniBatchKMeans
from torch.nn import functional as F

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger = logging

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask

def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def f0_to_coarse(f0):
  f0_mel = 1127 * (1 + f0 / 700).log()
  a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
  b = f0_mel_min * a - 1.
  f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
  # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
  f0_coarse = torch.round(f0_mel).long()
  f0_coarse = f0_coarse * (f0_coarse > 0)
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  f0_coarse = f0_coarse * (f0_coarse < f0_bin)
  f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
  return f0_coarse

def get_content(cmodel, y):
    with torch.no_grad():
        c = cmodel.extract_features(y.squeeze(1))[0]
    c = c.transpose(1, 2)
    return c

def get_f0_predictor(f0_predictor,hop_length,sampling_rate,**kargs):
    if f0_predictor == "pm":
        from modules.F0Predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "crepe":
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor
        f0_predictor_object = CrepeF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,device=kargs["device"],threshold=kargs["threshold"])
    elif f0_predictor == "harvest":
        from modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "dio":
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate) 
    elif f0_predictor == "rmvpe":
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor
        f0_predictor_object = RMVPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])
    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object

def get_speech_encoder(speech_encoder,device=None,**kargs):
    if speech_encoder == "vec768l12":
        from vencoder.ContentVec768L12 import ContentVec768L12
        speech_encoder_object = ContentVec768L12(device = device)
    elif speech_encoder == "vec256l9":
        from vencoder.ContentVec256L9 import ContentVec256L9
        speech_encoder_object = ContentVec256L9(device = device)
    elif speech_encoder == "vec256l9-onnx":
        from vencoder.ContentVec256L9_Onnx import ContentVec256L9_Onnx
        speech_encoder_object = ContentVec256L9_Onnx(device = device)
    elif speech_encoder == "vec256l12-onnx":
        from vencoder.ContentVec256L12_Onnx import ContentVec256L12_Onnx
        speech_encoder_object = ContentVec256L12_Onnx(device = device)
    elif speech_encoder == "vec768l9-onnx":
        from vencoder.ContentVec768L9_Onnx import ContentVec768L9_Onnx
        speech_encoder_object = ContentVec768L9_Onnx(device = device)
    elif speech_encoder == "vec768l12-onnx":
        from vencoder.ContentVec768L12_Onnx import ContentVec768L12_Onnx
        speech_encoder_object = ContentVec768L12_Onnx(device = device)
    elif speech_encoder == "hubertsoft-onnx":
        from vencoder.HubertSoft_Onnx import HubertSoft_Onnx
        speech_encoder_object = HubertSoft_Onnx(device = device)
    elif speech_encoder == "hubertsoft":
        from vencoder.HubertSoft import HubertSoft
        speech_encoder_object = HubertSoft(device = device)
    elif speech_encoder == "whisper-ppg":
        from vencoder.WhisperPPG import WhisperPPG
        speech_encoder_object = WhisperPPG(device = device)
    elif speech_encoder == "cnhubertlarge":
        from vencoder.CNHubertLarge import CNHubertLarge
        speech_encoder_object = CNHubertLarge(device = device)
    elif speech_encoder == "dphubert":
        from vencoder.DPHubert import DPHubert
        speech_encoder_object = DPHubert(device = device)
    elif speech_encoder == "whisper-ppg-large":
        from vencoder.WhisperPPGLarge import WhisperPPGLarge
        speech_encoder_object = WhisperPPGLarge(device = device)
    elif speech_encoder == "wavlmbase+":
        from vencoder.WavLMBasePlus import WavLMBasePlus
        speech_encoder_object = WavLMBasePlus(device = device)
    else:
        raise Exception("Unknown speech encoder")
    return speech_encoder_object 

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
              print("%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning." % k)
              logger.info("%s is not in the checkpoint" % k)
              new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
  """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  def name_key(_f):
      return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))
  def time_key(_f):
      return os.path.getmtime(os.path.join(path_to_models, _f))
  sort_key = time_key if sort_by_time else name_key
  def x_sorted(_x):
      return sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")], key=sort_key)
  to_del = [os.path.join(path_to_models, fn) for fn in
            (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
  def del_info(fn):
      return logger.info(f".. Free up space by deleting ckpt {fn}")
  def del_routine(x):
      return [os.remove(x), del_info(x)]
  [del_routine(fn) for fn in to_del]

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')

  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path, infer_mode = False):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)
  hparams =HParams(**config) if not infer_mode else InferHParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)



def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target


def mix_model(model_paths,mix_rate,mode):
  mix_rate = torch.FloatTensor(mix_rate)/100
  model_tem = torch.load(model_paths[0])
  models = [torch.load(path)["model"] for path in model_paths]
  if mode == 0:
     mix_rate = F.softmax(mix_rate,dim=0)
  for k in model_tem["model"].keys():
     model_tem["model"][k] = torch.zeros_like(model_tem["model"][k])
     for i,model in enumerate(models):
        model_tem["model"][k] += model[k]*mix_rate[i]
  torch.save(model_tem,os.path.join(os.path.curdir,"output.pth"))
  return os.path.join(os.path.curdir,"output.pth")
  
def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比 from RVC
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2.detach().cpu().numpy(), frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1).to(data2.device)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2).to(data2.device)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    )
    return data2

def train_index(spk_name,root_dir = "dataset/44k/"):  #from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    n_cpu = cpu_count()
    print("The feature index is constructing.")
    exp_dir = os.path.join(root_dir,spk_name)
    listdir_res = []
    for file in os.listdir(exp_dir):
       if ".wav.soft.pt" in file:
          listdir_res.append(os.path.join(exp_dir,file))
    if len(listdir_res) == 0:
        raise Exception("You need to run preprocess_hubert_f0.py!")
    npys = []
    for name in sorted(listdir_res):
        phone = torch.load(name)[0].transpose(-1,-2).numpy()
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            info = traceback.format_exc()
            print(info)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1] , "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    # faiss.write_index(
    #     index,
    #     f"added_{spk_name}.index"
    # )
    print("Successfully build index")
    return index


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def get(self,index):
    return self.__dict__.get(index)

  
class InferHParams(HParams):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = InferHParams(**v)
      self[k] = v

  def __getattr__(self,index):
    return self.get(index)


class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume
