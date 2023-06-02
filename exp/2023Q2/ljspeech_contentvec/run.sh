# docker run --ipc host --gpus all -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vc:v1 bash

conda activate sovits

python resample.py --sr2 44100 --in_dir dataset_raw/ljspeech --out_dir2 dataset/44k/ljspeech

python preprocess_flist_config.py \
    --speech_encoder vec768l12 \
    --source_dir dataset/44k/ljspeech \
    --output_dir filelists/ljspeech_contentvec 

# fix config.json and diffusion.yaml in filelists/ljspeech_contentvec

python preprocess_hubert_f0.py \
    --config_dir filelists/ljspeech_contentvec \
    --f0_predictor dio \
    --in_dir dataset/44k/ljspeech \
    --num_processes 8


CUDA_VISIBLE_DEVICES=4,5 python train.py -c filelists/ljspeech_contentvec/config.json -m ljspeech_contentvec