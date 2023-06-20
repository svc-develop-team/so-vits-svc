# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


mkdir -p filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams
grep 'SSB3003' filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/train.txt > filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
cat filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt.tmp >> filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/train.txt
grep 'SSB3003' filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/val.txt > filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/val.txt

cp filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams
# manually change the data list in config.json

mkdir -p logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams
cp logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_0.pth
cp logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/D_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/D_0.pth

CUDA_VISIBLE_DEVICES=4,5 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams



## test
CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_100000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_xiaolin

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_100000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_pride

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_100000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/ljspeech_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_ljspeech

CUDA_VISIBLE_DEVICES=6 python inference.py -m "logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_100000.pth" \
    -c "filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/wav_scp/xiaolin_en/wav_test.scp \
    --output_dir logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/SSB3003_xiaolin_en

