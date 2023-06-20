# docker run --ipc host --gpus all -v /home:/home -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python exp/2023Q2/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others/0614/prepare_yangjie.py

## resample and normalization
mkdir -p dataset_raw/yangjie/LYG0002
ln -s /home/tmp-yi.liu/yangjie/step5/LYG0002/*.wav dataset_raw/yangjie/LYG0002
python resample.py --sr2 44100 --in_dir dataset_raw/yangjie --out_dir2 dataset/44k/yangjie


## preprocess the training data
python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --source_dir dataset/44k/yangjie \
    --output_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie

# fix config.json and diffusion.yaml

CUDA_VISIBLE_DEVICES=7 python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie \
    --f0_predictor dio \
    --in_dir dataset/44k/yangjie \
    --num_processes 2


mv filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt.bak >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/train.txt


mkdir -p logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie
cp logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_0.pth
cp logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/D_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/D_0.pth

# manually fix speaker info in config.json
# n_speakers and spk

## train the base model
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie

## test
CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_55000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json" \
    -s LYG0002 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/xiaolin_en/wav_test.scp \
    --output_dir /home/tmp-yi.liu/yangjie/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/LYG0002_xiaolin_en

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_55000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json" \
    -s LYG0002 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/ljspeech_en/wav_test.scp \
    --output_dir /home/tmp-yi.liu/yangjie/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/LYG0002_ljspeech

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_55000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json" \
    -s LYG0002 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/multi_linguistic/wav_test.scp \
    --output_dir /home/tmp-yi.liu/yangjie/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/LYG0002_multilingual

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/G_55000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/config.json" \
    -s LYG0002 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/ted_v1/wav_test.scp \
    --output_dir /home/tmp-yi.liu/yangjie/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_yangjie/LYG0002_ted_v1