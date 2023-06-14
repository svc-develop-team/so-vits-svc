# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


mkdir -p filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker
grep 'SSB3000' filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/train.txt > filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker/train.txt
grep 'SSB3000' filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/val.txt > filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker/val.txt

cp filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker
# manually change the data list in config.json

mkdir -p logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker
ln -s ../G_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/G_0.pth
ln -s ../D_400000.pth logs/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_jams/D_0.pth

CUDA_VISIBLE_DEVICES=7 python train.py \
    -c filelists/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker/config.json \
    -m ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607_contentvec/tuning_databaker
