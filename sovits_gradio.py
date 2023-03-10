from inference.infer_tool_grad import VitsSvc
import gradio as gr
import os

class VitsGradio:
    def __init__(self):
        self.so = VitsSvc()
        self.lspk = []
        self.modelPaths = []
        for root,dirs,files in os.walk("checkpoints"):
            for dir in dirs:
                self.modelPaths.append(dir)
        with gr.Blocks() as self.Vits:
            with gr.Tab("VoiceConversion"):
                with gr.Row(visible=False) as self.VoiceConversion:
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                self.srcaudio = gr.Audio(label = "输入音频")
                                self.btnVC = gr.Button("说话人转换")
                            with gr.Column():
                                self.dsid = gr.Dropdown(label = "目标角色", choices = self.lspk)
                                self.tran = gr.Slider(label = "升降调", maximum = 60, minimum = -60, step = 1, value = 0)
                                self.th = gr.Slider(label = "切片阈值", maximum = 32767, minimum = -32768, step = 0.1, value = -40)
                        with gr.Row():
                            self.VCOutputs = gr.Audio()
                self.btnVC.click(self.so.inference, inputs=[self.srcaudio,self.dsid,self.tran,self.th], outputs=[self.VCOutputs])
            with gr.Tab("SelectModel"):
                with gr.Column():
                    modelstrs = gr.Dropdown(label = "模型", choices = self.modelPaths, value = self.modelPaths[0], type = "value")
                    devicestrs = gr.Dropdown(label = "设备", choices = ["cpu","cuda"], value = "cpu", type = "value")
                    btnMod = gr.Button("载入模型")
                    btnMod.click(self.loadModel, inputs=[modelstrs,devicestrs], outputs = [self.dsid,self.VoiceConversion])

    def loadModel(self, path, device):
        self.lspk = []
        self.so.set_device(device)
        self.so.loadCheckpoint(path)
        for spk, sid in self.so.hps.spk.items():
            self.lspk.append(spk)
        VChange = gr.update(visible = True)
        SDChange = gr.update(choices = self.lspk, value = self.lspk[0])
        return [SDChange,VChange]

grVits = VitsGradio()

grVits.Vits.launch()