# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


## data preparation

# ljspeech
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/LJSpeech
ln -s /nfs2/speech/data/tts/Datasets/LJSpeech-1.1/wavs/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/LJSpeech/

# vctk
python tools/vctk/prepare_data.py --type mic2 -i /nfs2/speech/data/tts/Datasets/VCTK/wav48_silence_trimmed -o dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/

# databaker
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3000
ln -s /nfs2/speech/data/tts/Datasets/DataBaker/Wave/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3000/

# aishell3
python tools/aishell3/prepare_data.py -i /nfs2/speech/data/tts/Datasets/aishell3 -o dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607

# mita
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3002
ln -s /nfs2/yi.liu/data/mita_20220830/Wave/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3002

# musha
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3001
ln -s /nfs2/speech/data/tts/andi_speech_haitian_20230510/processed/Wave_48K/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3001
# TODO: the former musha data is not used

# jams
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3003
ln -s /nfs2/speech/data/tts/jams/jams_haitian_202304_processed/misread/Wave_48K/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3003
ln -s /nfs2/speech/data/tts/jams/jams_haitian_202304_processed/video/Wave_48K/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3003
ln -s /nfs2/speech/data/tts/jams/jams_haitian_202304_processed/recording/Wave_48K/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB3003

# xiaolin
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/xiaolin
ln -s /nfs2/speech/data/tts/luster_e011_standard_female/MDT2020TTS04/WAV/*/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/xiaolin/
ln -s /nfs2/speech/data/tts/luster_e011_standard_female/*/wav/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/xiaolin

# zijian
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/zijian
ln -s /nfs1/yichao.hu/Datasets/LYG_zijian/IpaProsodyLabeling/Wave/*.wav dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/zijian

# chenguoping
mkdir -p dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/chenguoping
ls /nfs1/yichao.hu/Datasets/GWSK_ChenGuoPing/step3/IpaProsodyLabeling/Wave/*.wav |\
    awk -F '/' '{print "ln -s "$0" dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/chenguoping/"$NF}' | sh
ls /nfs1/yichao.hu/Datasets/GWSK_ChenGuoPing/step3/IpaProsodyLabelingWave3/Wave/*wav |\
    awk -F '/' '{print "ln -s "$0" dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/chenguoping/02"$NF}' | sh



## resample and normalization
python resample.py --sr2 44100 --in_dir dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 --out_dir2 dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607
cp -r dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large



## preprocess the training data
python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --source_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 \
    --output_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec

python preprocess_flist_config.py \
    --speech_encoder whisper-ppg-large \
    --source_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large \
    --output_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large


# fix config.json and diffusion.yaml

CUDA_VISIBLE_DEVICES=2 python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 \
    --num_processes 4

CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large \
    --num_processes 12



## train the base model
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec

CUDA_VISIBLE_DEVICES=2,3 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large


## test

# Databaker

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_xiaolin

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_jams

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_pride

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/ljspeech_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_ljspeech

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/xiaolin_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_xiaolin_en





# Jams
CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3003_xiaolin

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3003_jams

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3003_pride

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/ljspeech_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3003_ljspeech

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/xiaolin_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3003_xiaolin_en








# whisper
CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/G_225000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/jams

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/G_225000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/G_225000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/happy/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/happy

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/G_225000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_whisper_large/pride
