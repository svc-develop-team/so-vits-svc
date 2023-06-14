# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/databaker --out_dir2 dataset/44k/databaker

python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --decoder nsf_decoder \
    --source_dir dataset/44k/databaker \
    --output_dir filelists/databaker_contentvec

# fix config.json and diffusion.yaml in filelists/databaker_contentvec

CUDA_VISIBLE_DEVICES=0 python preprocess_hubert_f0.py \
    --config_dir filelists/databaker_contentvec \
    --f0_predictor dio \
    --in_dir dataset/44k/databaker \
    --num_processes 4

CUDA_VISIBLE_DEVICES=6,7 python train.py -c filelists/databaker_contentvec/config.json -m databaker_contentvec


# test

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_contentvec/G_230000.pth" -c "filelists/databaker_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/xiaolin/wav_test.scp --output_dir logs/databaker_contentvec/xiaolin

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_contentvec/G_230000.pth" -c "filelists/databaker_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/exp/fvae-vc/data/raw/jams/wav_test.scp --output_dir logs/databaker_contentvec/jams

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_contentvec/G_230000.pth" -c "filelists/databaker_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/happy/wav_test.scp --output_dir logs/databaker_contentvec/happy

CUDA_VISIBLE_DEVICES=1 python inference.py -m "logs/databaker_contentvec/G_230000.pth" -c "filelists/databaker_contentvec/config.json" \
    -s SSB3000 -f0p dio -a --slice_db -50 --clip 25 -lg 1 \
    --wav_scp /nfs2/guang.liang/datasets/magicdata_tts_train/MDT-TTS-G005/pride/wav_test.scp --output_dir logs/databaker_contentvec/pride





