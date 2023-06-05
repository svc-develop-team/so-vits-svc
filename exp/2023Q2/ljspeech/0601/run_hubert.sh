# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/ljspeech --out_dir2 dataset/44k/ljspeech_hubert

python preprocess_flist_config.py \
    --speech_encoder hubertsoft \
    --source_dir dataset/44k/ljspeech_hubert \
    --output_dir filelists/ljspeech_hubert

# fix config.json and diffusion.yaml in filelists/ljspeech_hubert

CUDA_VISIBLE_DEVICES=3 python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_hubert \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech_hubert \
    --num_processes 8


CUDA_VISIBLE_DEVICES=0,7 python train.py -c filelists/ljspeech_hubert/config.json -m ljspeech_hubert

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/ljspeech_hubert/G_40000.pth" -c "filelists/ljspeech_hubert/config.json" -s ljspeech -f0p dio -a --slice_db -100 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/ljspeech_hubert/test
