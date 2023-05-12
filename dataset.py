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

from typing import List, Dict, Any, Union, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
import datasets
from diffusers import DDPMScheduler
from functools import partial
import random

import numpy as np


@torch.no_grad()
def collate_fn(
        batch: List[Dict[str, Any]],
        noise_scheduler: DDPMScheduler,
        num_frames: int,
        hint_spacing: Optional[int] = None,
        as_numpy: bool = True
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    if hint_spacing is None or hint_spacing < 1:
        hint_spacing = num_frames
    if as_numpy:
        dtype = np.float32
    else:
        dtype = torch.float32
    prompts = []
    videos = []
    for s in batch:
        # prompt
        prompts.append(torch.tensor(s['prompt']).to(dtype = torch.float32))
        # frames
        frames = torch.tensor(s['video']).to(dtype = torch.float32)
        max_frames = len(frames)
        assert max_frames >= num_frames
        video_slice = random.randint(0, max_frames - num_frames)
        frames = frames[video_slice:video_slice + num_frames]
        frames = frames.permute(1, 0, 2, 3) # f, c, h, w -> c, f, h, w
        videos.append(frames)

    encoder_hidden_states = torch.cat(prompts) # b, 77, 768

    latents = torch.stack(videos) # b, c, f, h, w
    latents = latents * 0.18215
    hint_latents = latents[:, :, ::hint_spacing, :, :]
    hint_latents = hint_latents.repeat_interleave(hint_spacing, 2)
    #hint_latents = hint_latents[:, :, :num_frames-1, :, :]
    #input_latents = latents[:, :, 1:, :, :]
    input_latents = latents
    noise = torch.randn_like(input_latents)
    bsz = input_latents.shape[0]
    timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            dtype = torch.int64
    )
    noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
    mask = torch.zeros([
            noisy_latents.shape[0],
            1,
            noisy_latents.shape[2],
            noisy_latents.shape[3],
            noisy_latents.shape[4]
    ])
    latent_model_input = torch.cat([noisy_latents, mask, hint_latents], dim = 1)

    latent_model_input = latent_model_input.to(memory_format = torch.contiguous_format)
    encoder_hidden_states = encoder_hidden_states.to(memory_format = torch.contiguous_format)
    timesteps = timesteps.to(memory_format = torch.contiguous_format)
    noise = noise.to(memory_format = torch.contiguous_format)

    if as_numpy:
        latent_model_input = latent_model_input.numpy().astype(dtype)
        encoder_hidden_states = encoder_hidden_states.numpy().astype(dtype)
        timesteps = timesteps.numpy().astype(np.int32)
        noise = noise.numpy().astype(dtype)
    else:
        latent_model_input = latent_model_input.to(dtype = dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype = dtype)
        noise = noise.to(dtype = dtype)

    return {
            'latent_model_input': latent_model_input,
            'encoder_hidden_states': encoder_hidden_states,
            'timesteps': timesteps,
            'noise': noise
    }

def worker_init_fn(_: int):
    wseed = torch.initial_seed() % (2**32-2) # max val for random 2**32 - 1
    random.seed(wseed)
    np.random.seed(wseed)


def load_dataset(
        dataset_path: str,
        model_path: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 1,
        num_frames: int = 24,
        hint_spacing: Optional[int] = None,
        num_workers: int = 0,
        shuffle: bool = False,
        as_numpy: bool = True,
        pin_memory: bool = False,
        pin_memory_device: str = ''
) -> DataLoader:
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            model_path,
            subfolder = 'scheduler'
    )
    dataset = datasets.load_dataset(
            dataset_path,
            streaming = False,
            cache_dir = cache_dir
    )
    merged_dataset = ConcatDataset([ dataset[s] for s in dataset ])
    dataloader = DataLoader(
        merged_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        persistent_workers = num_workers > 0,
        drop_last = True,
        shuffle = shuffle,
        worker_init_fn = worker_init_fn,
        collate_fn = partial(collate_fn,
                noise_scheduler = noise_scheduler,
                num_frames = num_frames,
                hint_spacing = hint_spacing,
                as_numpy = as_numpy
        ),
        pin_memory = pin_memory,
        pin_memory_device = pin_memory_device
    )
    return dataloader


def validate_dataset(
        dataset_path: str
) -> List[int]:
    import os
    import json
    data_path = os.path.join(dataset_path, 'data')
    meta = set(os.path.splitext(x)[0] for x in os.listdir(os.path.join(data_path, 'metadata')))
    prompts = set(os.path.splitext(x)[0] for x in os.listdir(os.path.join(data_path, 'prompts')))
    videos = set(os.path.splitext(x)[0] for x in os.listdir(os.path.join(data_path, 'videos')))
    ok = meta.intersection(prompts).intersection(videos)
    all_of_em = meta.union(prompts).union(videos)
    not_ok = []
    for a in all_of_em:
        if a not in ok:
            not_ok.append(a)
    ok = list(ok)
    ok.sort()
    with open(os.path.join(data_path, 'id_list.json'), 'w') as f:
        json.dump(ok, f)

