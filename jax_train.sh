#!/bin/sh
# Make-A-Video Latent Diffusion Models
# Copyright (C) 2023  Lopho <contact@lopho.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#export WANDB_API_KEY="your_api_key"
export WANDB_ENTITY="tempofunk"
export WANDB_JOB_TYPE="train"
export WANDB_PROJECT="makeavid-sd-tpu"

python jax_train.py \
        --dataset_path ../storage/dataset/tempofunk-s \
        --model_path ../storage/trained_models/ep20 \
        --from_pt False \
        --convert2d False \
        --only_temporal True \
        --output_dir ../storage/output \
        --batch_size 1 \
        --num_frames 24 \
        --epochs 20 \
        --lr 0.00005 \
        --warmup 0.1 \
        --decay 0.0 \
        --sample_size 64 64 \
        --log_every_step 50 \
        --save_every_epoch 1 \
        --sample_every_epoch 1 \
        --seed 2 \
        --use_memory_efficient_attention True \
        --dtype bfloat16 \
        --param_dtype float32 \
        --verbose True \
        --dataset_cache_dir ../storage/cache/hf/datasets \
        --wandb True

# sudo rm /tmp/libtpu_lockfile

