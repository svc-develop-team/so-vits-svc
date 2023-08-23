import PySimpleGUI as sg
import sounddevice as sd
import torch, librosa, threading, pickle
from modules.enhancer import Enhancer
import numpy as np
from torch.nn import functional as F
from torchaudio.transforms import Resample
# from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
# from ddsp.core import upsample
import time
from gui_i18 import I18nAuto
from diffusion.infer_gt_mel import DiffGtMel
from inference.infer_tool import Svc
import os

class Config:
    def __init__(self) -> None:
        self.samplerate = 44100  # Hz
        self.block_time = 1.5  # s
        self.f_pitch_change: float = 0.0  # float(request_form.get("fPitchChange", 0))
        self.spk_id = 1  # 默认说话人。
        # self.spk_mix_dict = None  # {1:0.5, 2:0.5} 表示1号说话人和2号说话人的音色按照0.5:0.5的比例混合
        self.use_vocoder_based_enhancer = True
        self.use_feature_retrieval = False
        self.cluster_infer_ratio = 0
        self.checkpoint_path = ''
        self.kmeans_path = ''
        self.threhold = -35
        self.buffer_num = 2
        self.noice_scale = 0.4
        self.crossfade_time = 0.03
        self.select_pitch_extractor = 'fcpe'  # F0预测器["parselmouth", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
        # self.use_spk_mix = False
        self.sounddevices = ['', '']
        self.diff_use = False
        self.auto_F0 = False
        self.diff_project = ''
        self.diff_acc = 10
        self.diff_spk_id = 0
        self.k_step = 100
        self.diff_method = 'pndm'
        self.diff_silence = False
        self.second_encoding = False

    def save(self, path):
        with open(path + '\\config.pkl', 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, path) -> bool:
        try:
            with open(path + '\\config.pkl', 'rb') as f:
                self.update(pickle.load(f))
            return True
        except:
            print('config.pkl does not exist')
            return False
    
    def update(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

class GUI:
    def __init__(self) -> None:
        self.config = Config()
        self.flag_vc: bool = False  # 变声线程flag
        self.block_frame = 0
        self.crossfade_frame = 0
        self.sola_search_frame = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svc_model = None
        self.diff_model: DiffGtMel = DiffGtMel()
        self.fade_in_window: np.ndarray = None  # crossfade计算用numpy数组
        self.fade_out_window: np.ndarray = None  # crossfade计算用numpy数组
        self.input_wav: np.ndarray = None  # 输入音频规范化后的保存地址
        self.output_wav: np.ndarray = None  # 输出音频规范化后的保存地址
        self.sola_buffer: torch.Tensor = None  # 保存上一个output的crossfade
        self.f0_mode_list = ["pm", "dio", "harvest", "crepe" ,"rmvpe","fcpe"]  # F0预测器
        self.diff_method_list = ["ddim", "pndm", "dpm-solver++", "unipc"] # 加速采样方法
        self.f_safe_prefix_pad_length: float = 0.0
        self.resample_kernel = {}
        self.launcher()  # start

    def launcher(self):
        '''窗口加载'''
        input_devices, output_devices, _, _ = self.get_devices()
        sg.theme('LightBlue4')  # 设置主题
        sg.theme_background_color("#4BD2D8")
        sg.theme_element_background_color("#4BD2D8")
        sg.theme_text_element_background_color("#4BD2D8")
        # 界面布局
        layout = [
            [sg.Frame(layout=[
                [sg.Input(key='sg_model', default_text='logs\\44k\\G_30000.pth'),
                 sg.FileBrowse(i18n('选择模型文件'), key='choose_model')]
            ], title=i18n('模型：.pth格式(自动识别同目录下config.json)')),
                sg.Frame(layout=[
                    [sg.Text(i18n('选择配置文件所在目录')), sg.Input(key='config_file_dir', default_text='configs'),
                     sg.FolderBrowse(i18n('打开文件夹'), key='choose_config')],
                    [sg.Button(i18n('读取配置文件'), key='load_config'),
                     sg.Button(i18n('保存配置文件'), key='save_config')]
                ], title=i18n('快速配置文件'))
            ],
            [sg.Frame(layout=[
                [sg.Text(i18n("输入设备")),
                 sg.Combo(input_devices, key='sg_input_device', default_value=input_devices[sd.default.device[0]],
                          enable_events=True)],
                [sg.Text(i18n("输出设备")),
                 sg.Combo(output_devices, key='sg_output_device', default_value=output_devices[sd.default.device[1]],
                          enable_events=True)]
            ], title=i18n('音频设备')),
            sg.Frame(layout=[
                [sg.Input(key='kmeans_model', default_text='logs\\44k\\kmeans_10000.pt'),
                 sg.FileBrowse('选择聚类或特征检索文件', key='choose_model')]
            ], title="选择聚类或特征检索文件"),
            ],
            [sg.Frame(layout=[
                [sg.Text(i18n("说话人id")), sg.Input(key='spk_id', default_text='0', size=8)],
                [sg.Text(i18n("响应阈值")),
                 sg.Slider(range=(-60, 0), orientation='h', key='threhold', resolution=1, default_value=-45,
                           enable_events=True)],
                [sg.Text("特征检索/聚类比例"),
                 sg.Slider(range=(0, 1), orientation='h', key='cluster_infer_ratio', resolution=0.01, default_value=0,
                           enable_events=True)],
                [sg.Text(i18n("变调")),
                 sg.Slider(range=(-24, 24), orientation='h', key='pitch', resolution=1, default_value=0,
                           enable_events=True)],
                [sg.Text(i18n("采样率")), sg.Input(key='samplerate', default_text='44100', size=8)],
                [sg.Text("噪音级别，会影响咬字和音质"),
                 sg.Slider(range=(0, 1), orientation='h', key='noice_scale', resolution=0.01, default_value=0.4,
                           enable_events=True)],
                # [sg.Checkbox(text=i18n('启用捏音色功能'), default=False, key='spk_mix', enable_events=True),
                #  sg.Button(i18n("设置混合音色"), key='set_spk_mix')]
            ], title=i18n('普通设置')),
                sg.Frame(layout=[
                    [sg.Text(i18n("音频切分大小")),
                     sg.Slider(range=(0.05, 3.0), orientation='h', key='block', resolution=0.01, default_value=0.5,
                               enable_events=True)],
                    [sg.Text(i18n("交叉淡化时长")),
                     sg.Slider(range=(0.01, 0.15), orientation='h', key='crossfade', resolution=0.01,
                               default_value=0.04, enable_events=True)],
                    [sg.Text(i18n("使用历史区块数量")),
                     sg.Slider(range=(1, 20), orientation='h', key='buffernum', resolution=1, default_value=4,
                               enable_events=True)],
                    [sg.Text(i18n("f0预测模式")),
                     sg.Combo(values=self.f0_mode_list, key='f0_mode', default_value=self.f0_mode_list[-1],
                              enable_events=True)],
                    [sg.Checkbox(text=i18n('启用增强器'), default=False, key='use_enhancer', enable_events=True),
                     sg.Checkbox(text='启用特征检索', default=False, key='use_feature_retrieval', enable_events=True),
                     sg.Checkbox(text='自动F0预测', default=False, key='auto_F0', enable_events=True)
                    ],[
                        sg.Checkbox(text=i18n('不推理安全区(加速但损失效果)'), default=False, key='diff_silence', enable_events=True),
                    ]
                ], title=i18n('性能设置')),
                sg.Frame(layout=[
                    [sg.Text(i18n("扩散模型文件"))],
                    [sg.Input(key='diff_project', default_text='logs\\44k\\diffusion\\model_400000.pt'),
                     sg.FileBrowse(i18n('选择模型文件'), key='choose_model')],
                    [sg.Text(i18n("扩散说话人id")), sg.Input(key='diff_spk_id', default_text='1', size=18)],
                    [sg.Text(i18n("扩散深度")), sg.Input(key='k_step', default_text='100', size=18)],
                    [sg.Text(i18n("扩散加速")), sg.Input(key='diff_acc', default_text='10', size=18)],
                    [sg.Text(i18n("扩散算法")),
                     sg.Combo(values=self.diff_method_list, key='diff_method', default_value=self.diff_method_list[0],
                              enable_events=True)],
                    [sg.Checkbox(text=i18n('启用扩散'), key='diff_use', enable_events=True),
                     sg.Checkbox(text='启用二次编码', default=False, key='second_encoding', enable_events=True)
                     ]
                ], title=i18n('扩散设置')),
            ],
            [sg.Button(i18n("开始音频转换"), key="start_vc"), sg.Button(i18n("停止音频转换"), key="stop_vc"),
             sg.Text(i18n('推理所用时间(ms):')), sg.Text('0', key='infer_time')]
        ]

        # 创造窗口
        self.window = sg.Window('SOVITS - REAL - TIME - GUI', layout, finalize=True)
        self.window['spk_id'].bind('<Return>', '')
        self.window['samplerate'].bind('<Return>', '')
        self.window['diff_spk_id'].bind('<Return>', '')
        self.window['k_step'].bind('<Return>', '')
        self.window['diff_acc'].bind('<Return>', '')
        self.event_handler()

    def event_handler(self):
        '''事件处理'''
        while True:  # 事件处理循环
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:  # 如果用户关闭窗口
                self.flag_vc = False
                exit()
            
            print('event: ' + event)
            
            if event == 'start_vc' and self.flag_vc == False:
                # set values 和界面布局layout顺序一一对应
                self.set_values(values)
                print('crossfade_time:' + str(self.config.crossfade_time))
                print("buffer_num:" + str(self.config.buffer_num))
                print("samplerate:" + str(self.config.samplerate))
                print('block_time:' + str(self.config.block_time))
                print("prefix_pad_length:" + str(self.f_safe_prefix_pad_length))
                # print("mix_mode:" + str(self.config.spk_mix_dict))
                print("enhancer:" + str(self.config.use_vocoder_based_enhancer))
                print("diffusion:" + str(self.config.diff_use))
                print('using_cuda:' + str(torch.cuda.is_available()))
                self.start_vc()
            elif event == 'k_step':
                if 1 <= int(values['k_step']) <= 1000:
                    self.config.k_step = int(values['k_step'])
                else:
                    self.window['k_step'].update(1000)
            elif event == 'diff_acc':
                if self.config.k_step < int(values['diff_acc']):
                    self.config.diff_acc = int(self.config.k_step / 4)
                else:
                    self.config.diff_acc = int(values['diff_acc'])
            elif event == 'diff_spk_id':
                self.config.diff_spk_id = int(values['diff_spk_id'])
            elif event == 'diff_use':
                self.config.diff_use = values['diff_use']
                self.window['use_enhancer'].update(False)
                self.config.use_vocoder_based_enhancer=False
            elif event == 'diff_silence':
                self.config.diff_silence = values['diff_silence']
            elif event == 'diff_method':
                self.config.diff_method = values['diff_method']
            elif event == 'spk_id':
                self.config.spk_id = int(values['spk_id'])
            elif event == 'threhold':
                self.config.threhold = values['threhold']
            elif event == 'pitch':
                self.config.f_pitch_change = values['pitch']
            elif event == 'second_encoding':
                self.config.second_encoding = values['second_encoding']
            elif event == 'auto_F0':
                self.config.auto_F0 = values['auto_F0']
            elif event == 'noice_scale':
                self.config.noice_scale = values['noice_scale']
            # elif event == 'spk_mix':
            #     self.config.use_spk_mix = values['spk_mix']
            # elif event == 'set_spk_mix':
            #     spk_mix = sg.popup_get_text(message='示例：1:0.3,2:0.5,3:0.2', title="设置混合音色，支持多人")
            #     if spk_mix != None:
            #         self.config.spk_mix_dict = eval("{" + spk_mix.replace('，', ',').replace('：', ':') + "}")
            elif event == 'spk_mix':
                self.config.use_spk_mix = values['spk_mix']
            elif event == 'use_feature_retrieval':
                self.config.use_feature_retrieval = values['use_feature_retrieval']
            elif event == 'use_feature_retrieval':
                self.config.use_feature_retrieval = values['use_feature_retrieval']
            elif event == 'use_enhancer':
                self.config.use_vocoder_based_enhancer = values['use_enhancer']
                self.window['diff_use'].update(False)
                self.config.diff_use = False
            elif event == 'load_config' and self.flag_vc == False:
                if self.config.load(values['config_file_dir']):
                    self.update_values()
            elif event == 'save_config' and self.flag_vc == False:
                self.set_values(values)
                self.config.save(values['config_file_dir'])
            elif event != 'start_vc' and self.flag_vc == True:
                self.flag_vc = False

    def set_values(self, values):
        self.set_devices(values["sg_input_device"], values['sg_output_device'])
        self.config.sounddevices = [values["sg_input_device"], values['sg_output_device']]
        self.config.checkpoint_path = values['sg_model']
        self.config.spk_id = int(values['spk_id'])
        self.config.threhold = values['threhold']
        self.config.f_pitch_change = values['pitch']
        self.config.samplerate = int(values['samplerate'])
        self.config.block_time = float(values['block'])
        self.config.crossfade_time = float(values['crossfade'])
        self.config.second_encoding = values['second_encoding']
        self.config.buffer_num = int(values['buffernum'])
        self.config.select_pitch_extractor = values['f0_mode']
        self.config.use_vocoder_based_enhancer = values['use_enhancer']
        self.config.use_feature_retrieval = values['use_feature_retrieval']
        self.config.cluster_infer_ratio = values['cluster_infer_ratio']
        self.config.noice_scale = float(values['noice_scale'])
        self.config.kmeans_path = values['kmeans_model']
        # self.config.use_spk_mix = values['spk_mix']
        self.config.diff_use = values['diff_use']
        self.config.auto_F0 = values['auto_F0']
        self.config.diff_silence = values['diff_silence']
        self.config.diff_method = values['diff_method']
        self.config.diff_project = values['diff_project']
        self.config.diff_acc = int(values['diff_acc'])
        self.config.diff_spk_id = int(values['diff_spk_id'])
        self.config.k_step = int(values['k_step'])
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.last_delay_frame = int(0.02 * self.config.samplerate)
        self.input_frames = max(
            self.block_frame + self.crossfade_frame + self.sola_search_frame + 2 * self.last_delay_frame,
            (1 + self.config.buffer_num) * self.block_frame)
        self.f_safe_prefix_pad_length = self.config.block_time * self.config.buffer_num - self.config.crossfade_time - 0.01 - 0.02

    def update_values(self):
        self.window['sg_model'].update(self.config.checkpoint_path)
        self.window['sg_input_device'].update(self.config.sounddevices[0])
        self.window['sg_output_device'].update(self.config.sounddevices[1])
        self.window['spk_id'].update(self.config.spk_id)
        self.window['threhold'].update(self.config.threhold)
        self.window['pitch'].update(self.config.f_pitch_change)
        self.window['auto_F0'].update(self.config.auto_F0)
        self.window['samplerate'].update(self.config.samplerate)
        self.window['use_feature_retrieval'].update(self.config.use_feature_retrieval)
        self.window['cluster_infer_ratio'].update(self.config.cluster_infer_ratio)
        self.window['noice_scale'].update(self.config.noice_scale)
        self.window['kmeans_model'].update(self.config.kmeans_path)
        # self.window['spk_mix'].update(self.config.use_spk_mix)
        self.window['block'].update(self.config.block_time)
        self.window['crossfade'].update(self.config.crossfade_time)
        self.window['buffernum'].update(self.config.buffer_num)
        self.window['f0_mode'].update(self.config.select_pitch_extractor)
        self.window['use_enhancer'].update(self.config.use_vocoder_based_enhancer)
        self.window['diff_use'].update(self.config.diff_use)
        self.window['diff_silence'].update(self.config.diff_silence)
        self.window['diff_method'].update(self.config.diff_method)
        self.window['diff_project'].update(self.config.diff_project)
        self.window['diff_acc'].update(self.config.diff_acc)
        self.window['diff_spk_id'].update(self.config.diff_spk_id)
        self.window['k_step'].update(self.config.k_step)

    def start_vc(self):
        '''开始音频转换'''
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.input_wav = np.zeros(self.input_frames, dtype='float32')
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        self.fade_in_window = torch.sin(
            np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.update_model(self.config.checkpoint_path)
        if self.config.diff_use:
            self.diff_model.flush_model(self.config.diff_project, ddsp_config=self.svc_model.args)
        thread_vc = threading.Thread(target=self.soundinput)
        thread_vc.start()

    def soundinput(self):
        '''
        接受音频输入
        '''
        with sd.Stream(callback=self.audio_callback, blocksize=self.block_frame, samplerate=self.config.samplerate,
                       dtype='float32'):
            while self.flag_vc:
                time.sleep(self.config.block_time)
                print('Audio block passed.')
        print('ENDing VC')

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        '''
        音频处理
        '''
        start_time = time.perf_counter()
        print("\nStarting callback")
        self.input_wav[:] = np.roll(self.input_wav, -self.block_frame)
        self.input_wav[-self.block_frame:] = librosa.to_mono(indata.T)
        vol = self.svc_model.volume_extractor.extract(torch.FloatTensor(self.input_wav)[None,:].to(self.device))[None,:]
        vol_mask = (vol > 10 ** (float(self.config.threhold) / 20)).to(torch.float) #[1, T]
        vol_mask = torch.max_pool1d(vol_mask, kernel_size=8, stride=1, padding= 4)
        # infer
        _audio, _audio_len, n_frames = self.svc_model.infer(
            self.config.spk_id,
            self.config.f_pitch_change,
            self.input_wav,
            self.config.cluster_infer_ratio,
            self.config.auto_F0,
            self.config.noice_scale,
            False,
            self.config.select_pitch_extractor,
            0,
            0.05,
            self.config.k_step,
            0,
            False,
            self.config.second_encoding,
            1,
            vol
        )
        vol_mask = torch.nn.functional.interpolate(vol_mask[:,None,:], size=_audio.shape[-1], mode='linear')[0,0,:]
        _audio *= vol_mask

        _model_sr = self.svc_model.target_sample

        # debug sola
        '''
        _audio, _model_sr = self.input_wav, self.config.samplerate
        rs = int(np.random.uniform(-200,200))
        print('debug_random_shift: ' + str(rs))
        _audio = np.roll(_audio, rs)
        _audio = torch.from_numpy(_audio).to(self.device)
        '''

        if _model_sr != self.config.samplerate:
            key_str = str(_model_sr) + '_' + str(self.config.samplerate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(_model_sr, self.config.samplerate,
                                                         lowpass_filter_width=128).to(self.device)
            _audio = self.resample_kernel[key_str](_audio)
        temp_wav = _audio[
                   - self.block_frame - self.crossfade_frame - self.sola_search_frame - self.last_delay_frame: - self.last_delay_frame]

        # sola shift
        conv_input = temp_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_frame, device=self.device)) + 1e-8)
        sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        temp_wav = temp_wav[sola_shift: sola_shift + self.block_frame + self.crossfade_frame]
        print('sola_shift: ' + str(int(sola_shift)))

        temp_wav[: self.crossfade_frame] *= self.fade_in_window
        temp_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer = temp_wav[- self.crossfade_frame:]

        outdata[:] = temp_wav[: - self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        end_time = time.perf_counter()
        print('infer_time: ' + str(end_time - start_time))
        self.window['infer_time'].update(int((end_time - start_time) * 1000))

    def get_devices(self, update: bool = True):
        '''获取设备列表'''
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]
        return input_devices, output_devices, input_devices_indices, output_devices_indices

    def set_devices(self, input_device, output_device):
        '''设置输出设备'''
        input_devices, output_devices, input_device_indices, output_device_indices = self.get_devices()
        sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
        sd.default.device[1] = output_device_indices[output_devices.index(output_device)]
        print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
        print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))
    
    def update_model(self, model_path):
        model_dir = os.path.dirname(model_path)
        model_config_path = os.path.join(model_dir, 'config.json')
        model_diff = self.config.diff_project
        model_diff_dir = os.path.dirname(model_diff)
        model_diff_config_path = os.path.join(model_diff_dir, 'config.yaml')
        print("model_dir:",model_dir)
        print("model_config_path:",model_config_path)
        print("model_diff:",model_diff)
        print("model_diff_config_path:",model_diff_config_path)
        if os.path.exists(model_dir):
            self.svc_model = Svc(model_path,
                                model_config_path,
                                self.device,
                                cluster_model_path=self.config.kmeans_path,
                                nsf_hifigan_enhance=self.config.use_vocoder_based_enhancer,
                                diffusion_model_path=model_diff,
                                diffusion_config_path=model_diff_config_path,
                                shallow_diffusion=self.config.diff_use,
                                only_diffusion=False,
                                spk_mix_enable=False,
                                feature_retrieval=self.config.use_feature_retrieval,
                                )
            self.svc_model.net_g_ms.dec.onnx = True
            self.svc_model.net_g_ms.dec.m_source.l_sin_gen.onnx = True
            
if __name__ == "__main__":
    i18n = I18nAuto()
    gui = GUI()