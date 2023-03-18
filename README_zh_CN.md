# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

## 使用规约

1. 本项目是基于学术交流目的建立，仅供交流与学习使用，并非为生产环境准备，请自行解决数据集的授权问题，任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和一切后果！
2. 任何发布到视频平台的基于 sovits 制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频 / 音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
3. 由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例，注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！
4. 继续使用视为已同意本仓库 README 所述相关条例，本仓库 README 已进行劝导义务，不对后续可能存在问题负责。
5. 如将本仓库代码二次分发，或将由此项目产出的任何结果公开发表 (包括但不限于视频网站投稿)，请注明原作者及代码来源 (此仓库)。
6. 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者，十分感谢。

### 改善了交互的一个分支推荐：[34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

## update

> 更新了4.0-v2模型，全部流程同4.0，相比4.0在部分场景下有一定提升，但也有些情况有退步，具体可移步[4.0-v2分支](https://github.com/svc-develop-team/so-vits-svc/tree/4.0-v2)

## 模型简介

歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题

### 4.0版本更新内容

+ 特征输入更换为 [Content Vec](https://github.com/auspicious3000/contentvec) 
+ 采样率统一使用44100hz
+ 由于更改了hop size等参数以及精简了部分模型结构，推理所需显存占用**大幅降低**，4.0版本44khz显存占用甚至小于3.0版本的32khz
+ 调整了部分代码结构
+ 数据集制作、训练过程和3.0保持一致，但模型完全不通用，数据集也需要全部重新预处理
+ 增加了可选项 1：vc模式自动预测音高f0,即转换语音时不需要手动输入变调key，男女声的调能自动转换，但仅限语音转换，该模式转换歌声会跑调
+ 增加了可选项 2：通过kmeans聚类方案减小音色泄漏，即使得音色更加像目标音色

## 预先下载的模型文件

#### **必须项**

+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在`hubert`目录下

```shell
# contentvec
http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt
# 也可手动下载放在hubert目录
```

#### **可选项(强烈建议使用)**

+ 预训练底模文件： `G_0.pth` `D_0.pth`
  + 放在`logs/44k`目录下

从svc-develop-team(待定)或任何其他地方获取

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途

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

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)

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

### [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kv-3y2DmZo0uya8pEr1xk7cSB-4e_Pct?usp=sharing) [sovits4_for_colab.ipynb](https://colab.research.google.com/drive/1kv-3y2DmZo0uya8pEr1xk7cSB-4e_Pct?usp=sharing)

#### [23/03/16] 不再需要手动下载hubert

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

## 一些法律条例参考

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
