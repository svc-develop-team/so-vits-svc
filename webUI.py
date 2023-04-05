import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

import subprocess
import edge_tts
import asyncio
from scipy.io import wavfile
import librosa
import torch
import time

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
cuda = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cuda.append("cuda:{}".format(i))

def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling):
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
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling)
        model.clear_empty()
        os.remove(temp_path)
        #构建保存文件的路径，并保存到results文件夹内
        try:
            timestamp = str(int(time.time()))
            output_file = os.path.join("./results", sid + "_" + timestamp + ".wav")
            soundfile.write(output_file, _audio, model.target_sample, format="wav")
            return "Success", (model.target_sample, _audio)
        except Exception as e:
            return "自动保存失败，请手动保存，音乐输出见下", (model.target_sample, _audio)    
    except Exception as e:
        return "异常信息:"+str(e)+"\n请排障后重试",None
    
def tts_func(_text,_rate):
    #使用edge-tts把文字转成音频
    # voice = "zh-CN-XiaoyiNeural"#女性，较高音
    # voice = "zh-CN-YunxiNeural"#男性
    voice = "zh-CN-YunxiNeural"#男性
    output_file = _text[0:10]+".wav"
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate>=0:
        ratestr="+{:.0%}".format(_rate)
    elif _rate<0:
        ratestr="{:.0%}".format(_rate)#减号自带

    p=subprocess.Popen(["edge-tts",
                        "--text",_text,
                        "--write-media",output_file,
                        "--voice",voice,
                        "--rate="+ratestr]
                        ,shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
    p.wait() 
    return output_file

def vc_fn2(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,F0_mean_pooling):
    #使用edge-tts把文字转成音频
    output_file=tts_func(text2tts,tts_rate)

    #调整采样率
    sr2=44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2= text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2,sr2,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                )
    
    #读取音频
    sample_rate, data=gr_pu.audio_from_file(save_path2)
    vc_input=(sample_rate, data)

    a,b=vc_fn(sid, vc_input, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling)
    os.remove(output_file)
    os.remove(save_path2)
    return a,b

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
            device = gr.Dropdown(label="推理设备，默认为自动选择cpu和gpu",choices=["Auto",*cuda,"cpu"],value="Auto")
            gr.Markdown(value="""
                <font size=3>全部上传完毕后(全部文件模块显示download),点击模型解析进行解析：</font>
                """)
            model_analysis_button = gr.Button(value="模型解析")
            sid = gr.Dropdown(label="音色（说话人）")
            sid_output = gr.Textbox(label="Output Message")

            text2tts=gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
            tts_rate = gr.Number(label="tts语速", value=0)

            vc_input3 = gr.Audio(label="上传音频")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
            auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)
            F0_mean_pooling = gr.Checkbox(label="是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒/s", value=0)
            pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
            lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
            lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75,interactive=True)
            vc_submit = gr.Button("音频直接转换", variant="primary")
            vc_submit2 = gr.Button("文字转音频+转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
            def modelAnalysis(model_path,config_path,cluster_model_path,device):
                global model
                debug=False
                if debug:
                    model = Svc(model_path.name, config_path.name,device=device if device!="Auto" else None,cluster_model_path= cluster_model_path.name if cluster_model_path!=None else "")
                    spks = list(model.spk2id.keys())
                    device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
                    return sid.update(choices = spks,value=spks[0]),"ok,模型被加载到了设备{}之上".format(device_name)
                else:
                    try:
                        model = Svc(model_path.name, config_path.name,device=device if device!="Auto" else None,cluster_model_path= cluster_model_path.name if cluster_model_path!=None else "")
                        spks = list(model.spk2id.keys())
                        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
                        return sid.update(choices = spks,value=spks[0]),"ok,模型被加载到了设备{}之上".format(device_name)
                    except Exception as e:
                        return "","异常信息:"+str(e)+"\n请排障后重试"
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,F0_mean_pooling], [vc_output1, vc_output2])
        model_analysis_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device],[sid,sid_output])
    app.launch()


