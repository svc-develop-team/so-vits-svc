# Onnx 导出
使用 [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py)
+ 新建文件夹：`checkpoints` 并打开
+ 在 `checkpoints` 文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如 `aziplayer`
+ 将你的模型更名为 `model.pth`，配置文件更名为 `config.json`，并放置到刚才创建的 `aziplayer` 文件夹下
+ 将 [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py) 中 `path ="NyaruTaffy"`的`"NyaruTaffy"`修改为你的项目名称，`path = "aziplayer"`
+ 运行 [onnx_export.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/onnx_export.py) 
+ 等待执行完毕，在你的项目文件夹下会生成一个 `model.onnx`，即为导出的模型
# Onnx 模型支持的 UI
   + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ 注意：Hubert Onnx 模型请使用 MoeSS 提供的模型，目前无法自行导出（fairseq 中 Hubert 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题）
[Hubert4.0](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel)
