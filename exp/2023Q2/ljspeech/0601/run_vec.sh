# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/ljspeech --out_dir2 dataset/44k/ljspeech

python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --decoder vits_decoder \
    --source_dir dataset/44k/ljspeech \
    --output_dir filelists/ljspeech_contentvec 

# fix config.json and diffusion.yaml in filelists/ljspeech_contentvec

python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_contentvec \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech \
    --num_processes 8


CUDA_VISIBLE_DEVICES=4,5 python train.py -c filelists/ljspeech_contentvec/config.json -m ljspeech_contentvec


python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --decoder nsf_decoder \
    --source_dir dataset/44k/ljspeech \
    --output_dir filelists/ljspeech_contentvec_nsf

CUDA_VISIBLE_DEVICES=6,7 python train.py -c filelists/ljspeech_contentvec_nsf/config.json -m ljspeech_contentvec_nsf

# test

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec/G_195000.pth" -c "filelists/ljspeech_contentvec/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/vctk/wav_test.scp --output_dir logs/ljspeech_contentvec/vctk

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec/G_195000.pth" -c "filelists/ljspeech_contentvec/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/databaker/wav_test.scp --output_dir logs/ljspeech_contentvec/databaker

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec/G_195000.pth" -c "filelists/ljspeech_contentvec/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp --output_dir logs/ljspeech_contentvec/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec/G_195000.pth" -c "filelists/ljspeech_contentvec/config.json" -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/ljspeech_contentvec/jams


# test

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec_nsf/G_195000.pth" -c "filelists/ljspeech_contentvec_nsf/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/vctk/wav_test.scp --output_dir logs/ljspeech_contentvec_nsf/vctk

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec_nsf/G_195000.pth" -c "filelists/ljspeech_contentvec_nsf/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs1/yi.liu/src/fvae-vc/data/raw/databaker/wav_test.scp --output_dir logs/ljspeech_contentvec_nsf/databaker

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec_nsf/G_195000.pth" -c "filelists/ljspeech_contentvec_nsf/config.json" \
    -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp --output_dir logs/ljspeech_contentvec_nsf/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_contentvec_nsf/G_195000.pth" -c "filelists/ljspeech_contentvec_nsf/config.json" -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/ljspeech_contentvec_nsf/jams