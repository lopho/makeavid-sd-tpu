#!/bin/sh

export WANDB_ENTITY="tempofunk"
export WANDB_JOB_TYPE="train"
export WANDB_PROJECT="makeavid-sd-tpu"

        #--model_path ../storage/model \

python train.py \
        --dataset_path ../storage/dataset/tempofunk-s \
        --model_path ../storage/trained_models/ep10 \
        --from_pt False \
        --convert2d False \
        --output_dir ../storage/output \
        --batch_size 1 \
        --num_frames 24 \
        --epochs 10 \
        --lr 0.00005 \
        --warmup 0.1 \
        --decay 0.8 \
        --sample_size 64 64 \
        --log_every_step 50 \
        --save_every_epoch 1 \
        --sample_every_epoch 1 \
        --seed 1 \
        --use_memory_efficient_attention True \
        --dtype bfloat16 \
        --param_dtype float32 \
        --verbose True \
        --dataset_cache_dir ../storage/cache/hf/datasets \
        --wandb True \
        # lol

# sudo rm /tmp/libtpu_lockfile

