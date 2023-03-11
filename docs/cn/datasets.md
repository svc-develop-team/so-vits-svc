# 数据集准备
仅需要以以下文件结构将数据集放入 dataset_raw 目录即可
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

# 数据预处理
1. 重采样至 44100hz

```shell
python resample.py
```
2. 自动划分训练集 验证集 测试集 以及自动生成配置文件
```shell
python preprocess_flist_config.py
```
3. 生成 hubert 与 f0
```shell
python preprocess_hubert_f0.py
```
执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除 dataset_raw 文件夹了
