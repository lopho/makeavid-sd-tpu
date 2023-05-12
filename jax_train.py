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

import jax
_ = jax.device_count() # ugly hack to prevent tpu comms to lock/race or smth smh

from typing import Tuple, Optional
import os
from argparse import ArgumentParser

from makeavid_sd.flax_impl import FlaxTrainerUNetPseudo3D
from dataset import load_dataset

def train(
        dataset_path: str,
        model_path: str,
        output_dir: str,
        dataset_cache_dir: Optional[str] = None,
        from_pt: bool = True,
        convert2d: bool = False,
        only_temporal: bool = True,
        sample_size: Tuple[int, int] = (64, 64),
        lr: float = 5e-5,
        batch_size: int = 1,
        num_frames: int = 24,
        epochs: int = 10,
        warmup: float = 0.1,
        decay: float = 0.0,
        weight_decay: float = 1e-2,
        log_every_step: int = 50,
        save_every_epoch: int = 1,
        sample_every_epoch: int = 1,
        seed: int = 0,
        dtype: str = 'bfloat16',
        param_dtype: str = 'float32',
        use_memory_efficient_attention: bool = True,
        verbose: bool = True,
        use_wandb: bool = False
) -> None:
    log = lambda x: print(x) if verbose else None
    log('\n----------------')
    log('Init trainer')
    trainer = FlaxTrainerUNetPseudo3D(
            model_path = model_path,
            from_pt = from_pt,
            convert2d = convert2d,
            sample_size = sample_size,
            seed = seed,
            dtype = dtype,
            param_dtype = param_dtype,
            use_memory_efficient_attention = use_memory_efficient_attention,
            verbose = verbose,
            only_temporal = only_temporal
    )
    log('\n----------------')
    log('Init dataset')
    dataloader = load_dataset(
            dataset_path = dataset_path,
            model_path = model_path,
            cache_dir = dataset_cache_dir,
            batch_size = batch_size * trainer.num_devices,
            num_frames = num_frames,
            num_workers = min(trainer.num_devices * 2, os.cpu_count() - 1),
            as_numpy = True,
            shuffle = True
    )
    log('\n----------------')
    log('Train')
    if use_wandb:
        trainer.enable_wandb()
    trainer.train(
            dataloader = dataloader,
            epochs = epochs,
            num_frames = num_frames,
            log_every_step = log_every_step,
            save_every_epoch = save_every_epoch,
            sample_every_epoch = sample_every_epoch,
            lr = lr,
            warmup = warmup,
            decay = decay,
            weight_decay = weight_decay,
            output_dir = output_dir
    )
    log('\n----------------')
    log('Done')


if __name__ == '__main__':
    parser = ArgumentParser()
    bool_type = lambda x: x.lower() in ['true', '1', 'yes']
    parser.add_argument('-v', '--verbose', type = bool_type, default = True)
    parser.add_argument('-d', '--dataset_path', required = True)
    parser.add_argument('-m', '--model_path', required = True)
    parser.add_argument('-o', '--output_dir', required = True)
    parser.add_argument('-b', '--batch_size', type = int, default = 1)
    parser.add_argument('-f', '--num_frames', type = int, default = 24)
    parser.add_argument('-e', '--epochs', type = int, default = 2)
    parser.add_argument('--only_temporal', type = bool_type, default = True)
    parser.add_argument('--dataset_cache_dir', type = str, default = None)
    parser.add_argument('--from_pt', type = bool_type, default = True)
    parser.add_argument('--convert2d', type = bool_type, default = False)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--warmup', type = float, default = 0.1)
    parser.add_argument('--decay', type = float, default = 0.0)
    parser.add_argument('--weight_decay', type = float, default = 1e-2)
    parser.add_argument('--sample_size', type = int, nargs = 2, default = [64, 64])
    parser.add_argument('--log_every_step', type = int, default = 250)
    parser.add_argument('--save_every_epoch', type = int, default = 1)
    parser.add_argument('--sample_every_epoch', type = int, default = 1)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--use_memory_efficient_attention', type = bool_type, default = True)
    parser.add_argument('--dtype', choices = ['float32', 'bfloat16', 'float16'], default = 'bfloat16')
    parser.add_argument('--param_dtype', choices = ['float32', 'bfloat16', 'float16'], default = 'float32')
    parser.add_argument('--wandb', type = bool_type, default = False)
    args = parser.parse_args()
    args.sample_size = tuple(args.sample_size)
    if args.verbose:
        print(args)
    train(
            dataset_path = args.dataset_path,
            model_path = args.model_path,
            from_pt = args.from_pt,
            convert2d = args.convert2d,
            only_temporal = args.only_temporal,
            output_dir = args.output_dir,
            dataset_cache_dir = args.dataset_cache_dir,
            batch_size = args.batch_size,
            num_frames = args.num_frames,
            epochs = args.epochs,
            lr = args.lr,
            warmup = args.warmup,
            decay = args.decay,
            weight_decay = args.weight_decay,
            sample_size = args.sample_size,
            seed = args.seed,
            dtype = args.dtype,
            param_dtype = args.param_dtype,
            use_memory_efficient_attention = args.use_memory_efficient_attention,
            log_every_step = args.log_every_step,
            save_every_epoch = args.save_every_epoch,
            sample_every_epoch = args.sample_every_epoch,
            verbose = args.verbose,
            use_wandb = args.wandb
    )

