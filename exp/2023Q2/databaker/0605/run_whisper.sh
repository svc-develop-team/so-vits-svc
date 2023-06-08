# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker_whisper

python preprocess_flist_config.py \
    --speech_encoder whisper-ppg \
    --source_dir dataset/44k/databaker_whisper \
    --output_dir filelists/databaker_whisper

# fix config.json and diffusion.yaml in filelists/databaker_whisper

CUDA_VISIBLE_DEVICES=2 python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_whisper \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker_whisper \
    --num_processes 4

CUDA_VISIBLE_DEVICES=4,5 python train.py -c filelists/databaker_whisper/config.json -m databaker_whisper

#################

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker_whisper_large

python preprocess_flist_config.py \
    --speech_encoder whisper-ppg-large \
    --source_dir dataset/44k/databaker_whisper_large \
    --output_dir filelists/databaker_whisper_large

CUDA_VISIBLE_DEVICES=6 python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_whisper_large \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker_whisper_large \
    --num_processes 4

CUDA_VISIBLE_DEVICES=6,7 python train.py -c filelists/databaker_whisper_large/config.json -m databaker_whisper_large

# test

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_whisper_large/G_220000.pth" -c "filelists/databaker_whisper_large/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp --output_dir logs/databaker_whisper_large/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_whisper_large/G_220000.pth" -c "filelists/databaker_whisper_large/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/databaker_whisper_large/jams