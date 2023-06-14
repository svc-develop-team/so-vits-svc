# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits


## data
mkdir -p dataset/44k/databaker_aishell3_mita_musha_jams_others_0607
ln -s /nfs1/yi.liu/src/so-vits-svc/dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/{chenguoping,xiaolin,zijian} dataset/44k/databaker_aishell3_mita_musha_jams_others_0607
ln -s /nfs1/yi.liu/src/so-vits-svc/dataset/44k/ljspeech_vctk_databaker_aishell3_mita_musha_jams_others_0607/SSB* dataset/44k/databaker_aishell3_mita_musha_jams_others_0607


## preprocess
python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --source_dir dataset/44k/databaker_aishell3_mita_musha_jams_others_0607 \
    --output_dir filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec


## train
CUDA_VISIBLE_DEVICES=6,7 python train.py \
    -c filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json \
    -m databaker_aishell3_mita_musha_jams_others_0607_contentvec


## test
CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/jams

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/zijian/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/zijian

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/happy/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/happy

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3003 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/pride


CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/G_250000.pth" \
    -c "filelists/databaker_aishell3_mita_musha_jams_others_0607_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp \
    --output_dir logs/databaker_aishell3_mita_musha_jams_others_0607_contentvec/SSB3000_jams
