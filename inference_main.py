import logging

import soundfile

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='配置文件路径')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'], help='合成目标说话人名称')
    
    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="", help='聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", help='选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='是否使用角色融合')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例')

    # 浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')
    

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    if cluster_infer_ratio != 0:
        if args.cluster_model_path == "":
            if args.feature_retrieval:  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
                args.cluster_model_path = "logs/44k/feature_and_index.pkl"
            else:
                args.cluster_model_path = "logs/44k/kmeans_10000.pt"
    else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
        args.cluster_model_path = ""

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    infer_tool.mkdir(["raw", "results"])
    
    if len(spk_mix_map)<=1:
        use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]
    
    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            if only_diffusion :
                isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
