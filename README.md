# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

## Terms of Use

1. This project is established for academic exchange purposes only and is intended for communication and learning purposes. It is not intended for production environments. Please solve the authorization problem of the dataset on your own. You shall be solely responsible for any problems caused by the use of non-authorized datasets for training and all consequences thereof.
2. Any videos based on sovits that are published on video platforms must clearly indicate in the description that they are used for voice changing and specify the input source of the voice or audio, for example, using videos or audios published by others and separating the vocals as input source for conversion, which must provide clear original video or music links. If your own voice or other synthesized voices from other commercial vocal synthesis software are used as the input source for conversion, you must also explain it in the description.
3. You shall be solely responsible for any infringement problems caused by the input source. When using other commercial vocal synthesis software as input source, please ensure that you comply with the terms of use of the software. Note that many vocal synthesis engines clearly state in their terms of use that they cannot be used for input source conversion.
4. Continuing to use this project is deemed as agreeing to the relevant provisions stated in this repository README. This repository README has the obligation to persuade, and is not responsible for any subsequent problems that may arise.
5. If you distribute this repository's code or publish any results produced by this project publicly (including but not limited to video sharing platforms), please indicate the original author and code source (this repository).
6. If you use this project for any other plan, please contact and inform the author of this repository in advance. Thank you very much.

### A fork with a greatly improved interface：[34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

## Update

> Updated the 4.0-v2 model, the entire process is the same as 4.0. Compared to 4.0, there is some improvement in certain scenarios, but there are also some cases where it has regressed. Please refer to the [4.0-v2 branch](https://github.com/svc-develop-team/so-vits-svc/tree/4.0-v2) for more information.

## Model Introduction

The singing voice conversion model uses SoftVC content encoder to extract source audio speech features, then the vectors are directly fed into VITS instead of converting to a text based intermediate; thus the pitch and intonations are conserved. Additionally, the vocoder is changed to [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to solve the problem of sound interruption.

### 4.0 Version Update Content

- Feature input is changed to [Content Vec](https://github.com/auspicious3000/contentvec)
- The sampling rate is unified to use 44100Hz
- Due to the change of hop size and other parameters, as well as the streamlining of some model structures, the required GPU memory for inference is **significantly reduced**. The 44kHz GPU memory usage of version 4.0 is even smaller than the 32kHz usage of version 3.0.
- Some code structures have been adjusted
- The dataset creation and training process are consistent with version 3.0, but the model is completely non-universal, and the data set needs to be fully pre-processed again.
- Added an option 1: automatic pitch prediction for vc mode, which means that you don't need to manually enter the pitch key when converting speech, and the pitch of male and female voices can be automatically converted. However, this mode will cause pitch shift when converting songs.
- Added option 2: reduce timbre leakage through k-means clustering scheme, making the timbre more similar to the target timbre.

## Pre-trained Model Files

#### **Required**

- ContentVec: [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  - Place it under the `hubert` directory

```shell
# contentvec
wget -P hubert/ http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt
# Alternatively, you can manually download and place it in the hubert directory
```

#### **Optional(Strongly recommend)**

- Pre-trained model files: `G_0.pth` `D_0.pth`
  - Place them under the `logs/44k` directory

Get them from svc-develop-team(TBD) or anywhere else.

Although the pretrained model generally does not cause any copyright problems, please pay attention to it. For example, ask the author in advance, or the author has indicated the feasible use in the description clearly.

## Dataset Preparation

Simply place the dataset in the `dataset_raw` directory with the following file structure.

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

## Preprocessing

1. Resample to 44100hz

```shell
python resample.py
```

2. Automatically split the dataset into training, validation, and test sets, and generate configuration files

```shell
python preprocess_flist_config.py
```

3. Generate hubert and f0

```shell
python preprocess_hubert_f0.py
```

After completing the above steps, the dataset directory will contain the preprocessed data, and the dataset_raw folder can be deleted.

## Training

```shell
python train.py -c configs/config.json -m 44k
```

Note: During training, the old models will be automatically cleared and only the latest three models will be kept. If you want to prevent overfitting, you need to manually backup the model checkpoints, or modify the configuration file `keep_ckpts` to 0 to never clear them.

## Inference

Use [inference_main.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/inference_main.py)

Up to this point, the usage of version 4.0 (training and inference) is exactly the same as version 3.0, with no changes (inference now has command line support).

```shell
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

Required parameters:

- -m, --model_path: path to the model.
- -c, --config_path: path to the configuration file.
- -n, --clean_names: a list of wav file names located in the raw folder.
- -t, --trans: pitch adjustment, supports positive and negative (semitone) values.
- -s, --spk_list: target speaker name for synthesis.

Optional parameters: see the next section

- -a, --auto_predict_f0: automatic pitch prediction for voice conversion, do not enable this when converting songs as it can cause serious pitch issues.
- -cm, --cluster_model_path: path to the clustering model, fill in any value if clustering is not trained.
- -cr, --cluster_infer_ratio: proportion of the clustering solution, range 0-1, fill in 0 if the clustering model is not trained.

## Optional Settings

If the results from the previous section are satisfactory, or if you didn't understand what is being discussed in the following section, you can skip it, and it won't affect the model usage. (These optional settings have a relatively small impact, and they may have some effect on certain specific data, but in most cases, the difference may not be noticeable.)

### Automatic f0 prediction

During the 4.0 model training, an f0 predictor is also trained, which can be used for automatic pitch prediction during voice conversion. However, if the effect is not good, manual pitch prediction can be used instead. But please do not enable this feature when converting singing voice as it may cause serious pitch shifting!

- Set "auto_predict_f0" to true in inference_main.

### Cluster-based timbre leakage control

Introduction: The clustering scheme can reduce timbre leakage and make the trained model sound more like the target's timbre (although this effect is not very obvious), but using clustering alone will lower the model's clarity (the model may sound unclear). Therefore, this model adopts a fusion method to linearly control the proportion of clustering and non-clustering schemes. In other words, you can manually adjust the ratio between "sounding like the target's timbre" and "being clear and articulate" to find a suitable trade-off point.

The existing steps before clustering do not need to be changed. All you need to do is to train an additional clustering model, which has a relatively low training cost.

- Training process:
  - Train on a machine with a good CPU performance. According to my experience, it takes about 4 minutes to train each speaker on a Tencent Cloud 6-core CPU.
  - Execute "python cluster/train_cluster.py". The output of the model will be saved in "logs/44k/kmeans_10000.pt".
- Inference process:
  - Specify "cluster_model_path" in inference_main.
  - Specify "cluster_infer_ratio" in inference_main, where 0 means not using clustering at all, 1 means only using clustering, and usually 0.5 is sufficient.

### [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kv-3y2DmZo0uya8pEr1xk7cSB-4e_Pct?usp=sharing) [sovits4_for_colab.ipynb](https://colab.research.google.com/drive/1kv-3y2DmZo0uya8pEr1xk7cSB-4e_Pct?usp=sharing)

#### [23/03/16] No longer need to download hubert manually

## Exporting to Onnx

Use [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)

- Create a folder named `checkpoints` and open it
- Create a folder in the `checkpoints` folder as your project folder, naming it after your project, for example `aziplayer`
- Rename your model as `model.pth`, the configuration file as `config.json`, and place them in the `aziplayer` folder you just created
- Modify `"NyaruTaffy"` in `path = "NyaruTaffy"` in [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py) to your project name, `path = "aziplayer"`
- Run [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)
- Wait for it to finish running. A `model.onnx` will be generated in your project folder, which is the exported model.

### UI support for Onnx models

- [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)

Note: For Hubert Onnx models, please use the models provided by MoeSS. Currently, they cannot be exported on their own (Hubert in fairseq has many unsupported operators and things involving constants that can cause errors or result in problems with the input/output shape and results when exported.)  [Hubert4.0](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel)

## Some legal provisions for reference

#### 《民法典》

##### 第一千零一十九条 

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。
未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。
对自然人声音的保护，参照适用肖像权保护的有关规定。

#####  第一千零二十四条 

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。  

#####  第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。
行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。  
