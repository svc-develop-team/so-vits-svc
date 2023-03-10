# SoftVC VITS Singing Voice Conversion

## Updates
> According to incomplete statistics, it seems that training with multiple speakers may lead to **worsened leaking of voice timbre**. It is not recommended to train models with more than 5 speakers. The current suggestion is to try to train models with only a single speaker if you want to achieve a voice timbre that is more similar to the target.
> Fixed the issue with unwanted staccato, improving audio quality by a decent amount.\
> The 2.0 version has been moved to the 2.0 branch.\
> Version 3.0 uses the code structure of FreeVC, which isn't compatible with older versions.\
> Compared to [DiffSVC](https://github.com/prophesier/diff-svc) , diffsvc performs much better when the training data is of extremely high quality, but this repository may perform better on datasets with lower quality. Additionally, this repository is much faster in terms of inference speed compared to diffsvc.

## Model Overview
A singing voice conversion (SVC) model, using the SoftVC encoder to extract features from the input audio, sent into VITS along with the F0 to replace the original input to achieve a voice conversion effect. Additionally, changing the vocoder to [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to fix the issue with unwanted staccato.

## Notice
+ The current branch is the 32kHz version, which requires less vram during inferencing, as well as faster inferencing speeds, and datasets for said branch take up less disk space. Thus the 32 kHz branch is recommended for use.
+ If you want to train 48 kHz variant models, switch to the [main branch](https://github.com/innnky/so-vits-svc/tree/main).


## Required models
+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + Place under `hubert`.
+ Pretrained models [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth) and [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth)
  + Place under `logs/32k`.
  + Pretrained models are required, because from experiments, training from scratch can be rather unpredictable to say the least, and training with a pretrained model can greatly improve training speeds.
  + The pretrained model includes云灏, 即霜, 辉宇·星AI, 派蒙, and 绫地宁宁, covering the common ranges of both male and female voices, and so it can be seen as a rather universal pretrained model.
  + The pretrained model exludes the `optimizer speaker_embedding` section, rendering it only usable for pretraining and incapable of inferencing with.
```shell
# For simple downloading.
# hubert
wget -P hubert/ https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
# G&D pretrained models
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth

```

## Colab notebook script for dataset creation and training.
[colab training notebook](https://colab.research.google.com/drive/1rCUOOVG7-XQlVZuWRAj5IpGrMM8t07pE?usp=sharing)

## Dataset preparation
All that is required is that the data be put under the `dataset_raw` folder in the structure format provided below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Data pre-processing.
1. Resample to 32khz

```shell
python resample.py
 ```
2. Automatically sort out training set, validation set, test set, and automatically generate configuration files.
```shell
python preprocess_flist_config.py
# Notice.
# The n_speakers value in the config will be set automatically according to the amount of speakers in the dataset.
# To reserve space for additionally added speakers in the dataset, the n_speakers value will be be set to twice the actual amount.
# If you want even more space for adding more data, you can edit the n_speakers value in the config after runing this step.
# This can not be changed after training starts.
```
3. Generate hubert and F0 features/
```shell
python preprocess_hubert_f0.py
```
After running the step above, the `dataset` folder will contain all the pre-processed data, you can delete the `dataset_raw` folder after that.

## Training.
```shell
python train.py -c configs/config.json -m 32k
```

## Inferencing.

Use [inference_main.py](inference_main.py)
+ Edit `model_path` to your newest checkpoint.
+ Place the input audio under the `raw` folder.
+ Change `clean_names` to the output file name.
+ Use `trans` to edit the pitch shifting amount (semitones). 
+ Change `spk_list` to the speaker name.

## Onnx Exporting.
### **When exporting Onnx, please make sure you re-clone the whole repository!!!**
Use [onnx_export.py](onnx_export.py)
+ Create a new folder called `checkpoints`.
+ Create a project folder in `checkpoints` folder with the desired name for your project, let's use `myproject` as example. Folder structure looks like `./checkpoints/myproject`.
+ Rename your model to `model.pth`, rename your config file to `config.json` then move them into `myproject` folder.
+ Modify [onnx_export.py](onnx_export.py) where `path = "NyaruTaffy"`, change `NyaruTaffy` to your project name, here it will be `path = "myproject"`.
+ Run [onnx_export.py](onnx_export.py)
+ Once it finished, a `model.onnx` will be generated in `myproject` folder, that's the model you just exported.
+ Notice: if you want to export a 48K model, please follow the instruction below or use `model_onnx_48k.py` directly.
    + Open [model_onnx.py](model_onnx.py) and change `hps={"sampling_rate": 32000...}` to `hps={"sampling_rate": 48000}` in class `SynthesizerTrn`.
    + Open [nvSTFT](/vdecoder/hifigan/nvSTFT.py) and replace all `32000` with `48000`
    ### Onnx Model UI Support
    + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ All training function and transformation are removed, only if they are all removed you are actually using Onnx.

## Gradio (WebUI)
Use [sovits_gradio.py](sovits_gradio.py) to run Gradio WebUI
+ Create a new folder called `checkpoints`.
+ Create a project folder in `checkpoints` folder with the desired name for your project, let's use `myproject` as example. Folder structure looks like `./checkpoints/myproject`.
+ Rename your model to `model.pth`, rename your config file to `config.json` then move them into `myproject` folder.
+ Run [sovits_gradio.py](sovits_gradio.py)
