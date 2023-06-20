# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


## data preparation

## resample and normalization
python resample.py --sr2 16000 --in_dir dataset_raw/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607 --out_dir2 dataset/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619


## preprocess the training data
python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --configs_template ./configs_template_16k \
    --source_dir dataset/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619 \
    --output_dir filelists/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619_contentvec



# fix config.json and diffusion.yaml

CUDA_VISIBLE_DEVICES=0,1,4,5 python preprocess_hubert_f0.py \
    --config_dir filelists/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619_contentvec \
    --f0_predictor dio \
    --in_dir dataset/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619 \
    --num_processes 12



## train the base model
CUDA_VISIBLE_DEVICES=2,3,4,5 python train.py \
    -c filelists/16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619_contentvec/config.json \
    -m 16k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0619_contentvec





















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




