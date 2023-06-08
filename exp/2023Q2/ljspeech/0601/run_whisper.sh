# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/ljspeech --out_dir2 dataset/44k/ljspeech_whisper

python preprocess_flist_config.py \
    --speech_encoder whisper-ppg \
    --source_dir dataset/44k/ljspeech_whisper \
    --output_dir filelists/ljspeech_whisper

# fix config.json and diffusion.yaml in filelists/ljspeech_whisper

CUDA_VISIBLE_DEVICES=3 python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_whisper \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech_whisper \
    --num_processes 8


CUDA_VISIBLE_DEVICES=6,7 python train.py -c filelists/ljspeech_whisper/config.json -m ljspeech_whisper


# test

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_whisper/G_230000.pth" -c "filelists/ljspeech_whisper/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/vctk/wav_test.scp --output_dir logs/ljspeech_whisper/vctk

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_whisper/G_230000.pth" -c "filelists/ljspeech_whisper/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/databaker/wav_test.scp --output_dir logs/ljspeech_whisper/databaker

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_whisper/G_230000.pth" -c "filelists/ljspeech_whisper/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp --output_dir logs/ljspeech_whisper/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_whisper/G_230000.pth" -c "filelists/ljspeech_whisper/config.json" -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/ljspeech_whisper/jams