import io
import logging

import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import RealTimeVC, Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())

    # 模型推理
    if raw_infer:
        # out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                            auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 启用则为直接切片合成，False为交叉淡化方式
    # vst插件调整0.3-0.5s切片时间可以降低延迟，直接切片方法会有连接处爆音、交叉淡化会有轻微重叠声音
    # 自行选择能接受的方法，或将vst最大切片时间调整为1s，此处设为Ture，延迟大音质稳定一些
    raw_infer = True
    # 每个模型和config是唯一对应的
    model_name = "logs/32k/G_174000-Copy1.pth"
    config_name = "configs/config.json"
    cluster_model_path = "logs/44k/kmeans_10000.pt"
    svc_model = Svc(model_name, config_name, cluster_model_path=cluster_model_path)
    svc = RealTimeVC()
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
