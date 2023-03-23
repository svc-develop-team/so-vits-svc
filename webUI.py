import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None

def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, model.target_sample, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,pad_seconds,cl_num,lg_num,lgr_num)
        model.clear_empty()
        os.remove(temp_path)
        return "Success", (model.target_sample, _audio)
    except Exception as e:
        return "异常信息:"+str(e)+"\n请排障后重试",None

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Sovits4.0"):
            gr.Markdown(value="""
                Sovits4.0 WebUI
                """)
            
            gr.Markdown(value="""
                <font size=3>下面是模型文件选择：</font>
                """)
            model_path = gr.File(label="模型文件")
            gr.Markdown(value="""
                <font size=3>下面是配置文件选择：</font>
                """)
            config_path = gr.File(label="配置文件")
            gr.Markdown(value="""
                <font size=3>下面是聚类模型文件选择，没有可以不填：</font>
                """)
            cluster_model_path = gr.File(label="聚类模型文件")
            device = gr.Dropdown(label="推理设备，留白则为自动选择cpu和gpu",choices=[None,"gpu","cpu"],value=None)
            gr.Markdown(value="""
                <font size=3>全部上传完毕后(全部文件模块显示download),点击模型解析进行解析：</font>
                """)
            model_analysis_button = gr.Button(value="模型解析")
            sid = gr.Dropdown(label="音色（说话人）")
            sid_output = gr.Textbox(label="Output Message")
            vc_input3 = gr.Audio(label="上传音频")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
            auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒/s", value=0)
            pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
            lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
            lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75,interactive=True)
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
            def modelAnalysis(model_path,config_path,cluster_model_path,device):
                try:
                    global model
                    model = Svc(model_path.name, config_path.name,device=device if device!="" else None,cluster_model_path= cluster_model_path.name if cluster_model_path!=None else "")
                    spks = list(model.spk2id.keys())
                    return sid.update(choices = spks,value=spks[0]),"ok"
                except Exception as e:
                    return "","异常信息:"+str(e)+"\n请排障后重试"
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num], [vc_output1, vc_output2])
        model_analysis_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device],[sid,sid_output])
    app.launch()



