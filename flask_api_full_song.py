import io

import numpy as np
import soundfile
from flask import Flask, request, send_file

from inference import infer_tool, slicer

app = Flask(__name__)


@app.route("/wav2wav", methods=["POST"])
def wav2wav():
    request_form = request.form
    audio_path = request_form.get("audio_path", None)  # wav文件地址
    tran = int(float(request_form.get("tran", 0)))  # 音调
    spk = request_form.get("spk", 0)  # 说话人(id或者name都可以,具体看你的config)
    wav_format = request_form.get("wav_format", 'wav')  # 范围文件格式
    infer_tool.format_wav(audio_path)
    chunks = slicer.cut(audio_path, db_thresh=-40)
    audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * 0.5)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
            svc_model.clear_empty()
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * 0.5)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name=f"temp.{wav_format}", as_attachment=True)


if __name__ == '__main__':
    model_name = "logs/44k/G_60000.pth"  # 模型地址
    config_name = "configs/config.json"  # config地址
    svc_model = infer_tool.Svc(model_name, config_name)
    app.run(port=1145, host="0.0.0.0", debug=False, threaded=False)
