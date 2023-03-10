# SoftVC VITS Singing Voice Conversion

## 使用规约
1. 请自行解决数据集的授权问题，任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和一切后果，与sovits无关！
2. 任何发布到视频平台的基于sovits制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频/音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
3. 由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例，注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！


## 模型简介
歌声音色转换模型，使用[Content Vec](https://github.com/auspicious3000/contentvec) 提取内容特征，输入visinger2模型合成目标声音

### 4.0 v2版本更新内容
+ 模型架构完全修改成[visinger2](https://github.com/zhangyongmao/VISinger2) 架构
+ 其他和4.0完全一致
### 4.0 v2版本特点
+ 在部分场景下比4.0有一定提升（例如部分场景的呼吸音电流音问题）
+ 但也有部分场景效果也有一定倒退，例如在猫雷数据上训练出来效果并不如4.0，而且在部分情况会合成出很鬼畜的声音
+ 至于炼老的还是v2 可以自己尝试下面的demo和4.0分支上的demo后对比决定
+ 4.0-v2是sovits的最后一个版本，之后不会再有更新，在基本验证没有大的bug后sovits即将Archive

在线demo：[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/innnky/sovits4.0-v2)

## 注意
+ 4.0-v2全部流程与4.0相同，环境与4.0相同，4.0预处理完成的数据和环境可以直接用
+ 与4.0不同的地方在于：
  + 模型**完全** 不通用，旧模型不可使用，底模也需要使用全新的底模, 请确保你加载了正确的底模否则训练时间会究极长！
  + config文件结构很不一样，不要使用老的config，如果是使用4.0的数据集则只需要执行preprocess_flist_config.py这一步生成新的config

## 预先下载的模型文件
+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在`hubert`目录下
+ 预训练底模文件： [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/G_0.pth) 与 [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/D_0.pth)
  + 放在`logs/44k`目录下
  + 预训练底模训练数据集覆盖男女生常见音域，可以认为是相对通用的底模
```shell
# 一键下载
# contentvec
# 由于作者提供的网盘没有直链，所以需要手动下载放在hubert目录
# G与D预训练模型:
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/G_0.pth
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/D_0.pth

```

[//]: # (## colab一键数据集制作、训练脚本)

[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/19fxpo-ZoL_ShEUeZIZi6Di-YioWrEyhR#scrollTo=0gQcIZ8RsOkn&#41;)

后面部分的readme和4.0一样了，没有变化

## 数据集准备
仅需要以以下文件结构将数据集放入dataset_raw目录即可
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


## 数据预处理
1. 重采样至 44100hz

```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集 以及自动生成配置文件
```shell
python preprocess_flist_config.py
```
3. 生成hubert与f0
```shell
python preprocess_hubert_f0.py
```
执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹了


## 训练
```shell
python train.py -c configs/config.json -m 44k
```
注：训练时会自动清除老的模型，只保留最新3个模型，如果想防止过拟合需要自己手动备份模型记录点,或修改配置文件keep_ckpts 0为永不清除

## 推理
使用 [inference_main.py](inference_main.py)

截止此处，4.0使用方法（训练、推理）和3.0完全一致，没有任何变化（推理增加了命令行支持）

```shell
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```
必填项部分
+ -m, --model_path：模型路径。
+ -c, --config_path：配置文件路径。
+ -n, --clean_names：wav 文件名列表，放在 raw 文件夹下。
+ -t, --trans：音高调整，支持正负（半音）。
+ -s, --spk_list：合成目标说话人名称。

可选项部分：见下一节
+ -a, --auto_predict_f0：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调。
+ -cm, --cluster_model_path：聚类模型路径，如果没有训练聚类则随便填。
+ -cr, --cluster_infer_ratio：聚类方案占比，范围 0-1，若没有训练聚类模型则填 0 即可。

## 可选项
如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用。(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)，
### 自动f0预测
4.0模型训练过程会训练一个f0预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！
+ 在inference_main中设置auto_predict_f0为true即可
### 聚类音色泄漏控制
介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，
可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低
+ 训练过程：
  + 使用cpu性能较好的机器训练，据我的经验在腾讯云6核cpu训练每个speaker需要约4分钟即可完成训练
  + 执行python cluster/train_cluster.py ，模型的输出会在 logs/44k/kmeans_10000.pt
+ 推理过程：
  + inference_main中指定cluster_model_path
  + inference_main中指定cluster_infer_ratio，0为完全不使用聚类，1为只使用聚类，通常设置0.5即可

## Onnx导出
使用 [onnx_export.py](onnx_export.py)
+ 新建文件夹：`checkpoints` 并打开
+ 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
+ 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
+ 将 [onnx_export.py](onnx_export.py) 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer"`
+ 运行 [onnx_export.py](onnx_export.py) 
+ 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型
   ### Onnx模型支持的UI
   + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ 我去除了所有的训练用函数和一切复杂的转置，一行都没有保留，因为我认为只有去除了这些东西，才知道你用的是Onnx
+ 注意：Hubert Onnx模型请使用MoeSS提供的模型，目前无法自行导出（fairseq中Hubert有不少onnx不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出shape和结果都有问题）
[Hubert4.0](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel)
