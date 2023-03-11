# 预先下载的模型文件

sovits 依赖于以下模型进行训练、推理

+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在 `hubert` 目录下
+ 预训练底模文件（仅训练，可选）： [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth) 与 [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth)
  + 放在 `logs/44k` 目录下
```shell
# 一键下载
# contentvec
wget -P hubert/http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt
# 也可手动下载放在 hubert 目录
# G 与 D 预训练模型:
wget -P logs/44k/https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth
wget -P logs/44k/https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth

```