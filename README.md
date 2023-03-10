# SoftVC VITS Singing Voice Conversion
## 重大BUG修复
+ 断音问题已解决，音质提升了一个档次
+ 过几天更新

## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。
> 目前模型修使用 [coarse F0](https://github.com/PlayVoice/VI-SVC/blob/main/svc/prepare/preprocess_wave.py) ，尝试使用[HarmoF0](https://github.com/wx-wei/harmof0) 进行f0提取但效果不佳，尝试使用[icassp2022-vocal-transcription](https://github.com/keums/icassp2022-vocal-transcription)提取midi替换f0输入但效果不佳

模型推理、训练、一键脚本汇总整理仓库 [sovits_guide](https://github.com/IceKyrin/sovits_guide)

