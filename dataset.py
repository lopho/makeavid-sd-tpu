import torch
from torch.utils.data import DataLoader, ChainDataset
import datasets
from transformers import CLIPTextModel
from diffusers import DDPMScheduler
from functools import partial
import random
from typing import List, Dict, Any

# min frames = 129

def collate_fn(
        batch: List[Dict[str, Any]],
        encoder: CLIPTextModel,
        noise_scheduler: DDPMScheduler,
        num_frames: int,
        hint_spacing: int = 0,
        dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    if hint_spacing < 1:
        hint_spacing = num_frames
    prompts = []
    videos = []
    for s in batch:
        # prompt
        tokens = s['tokenized_prompt']
        prompt = encoder(input_ids = torch.tensor(tokens['input_ids'])).last_hidden_state.squeeze()
        prompts.append(prompt)
        # frames
        frames = s['video']
        max_frames = len(frames)
        assert max_frames >= num_frames
        video_slice = random.randint(0, max_frames - num_frames)
        frames = frames[video_slice:video_slice + num_frames]
        output_frames = []
        for f in frames:
            mean = torch.tensor(f['mean'])
            std = torch.tensor(f['std'])
            output_frames.append(mean + std * torch.randn_like(mean))
        output_frames = torch.stack(output_frames).permute(1, 0, 2, 3) # f, c, h, w -> c, f, h, w
        videos.append(output_frames)

    encoder_hidden_states = torch.stack(prompts) # b, 77, 768

    latents = torch.stack(videos) # b, c, f, h, w
    latents = latents * 0.18215
    hint_latents = latents[:, :, 0::hint_spacing, :, :]
    hint_latents = hint_latents.repeat_interleave(hint_spacing, 2)
    hint_latents = hint_latents[:,:,:num_frames-1,:,:]
    input_latents = latents[:, :, 1:, :, :]
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

    return {
            'latent_model_input': latent_model_input.to(dtype = dtype, memory_format = torch.contiguous_format),
            'encoder_hidden_states': encoder_hidden_states.to(dtype = dtype,  memory_format = torch.contiguous_format),
            'timesteps': timesteps.to(memory_format = torch.contiguous_format),
            'noise': noise.to(dtype = dtype, memory_format = torch.contiguous_format)
    }


def load_dataset(
        dataset_path: str,
        pretrained: str = 'lxj616/make-a-stable-diffusion-video-timelapse',
        batch_size: int = 1,
        num_frames: int = 25,
        hint_spacing: int = 0,
        num_workers: int = 0,
        dtype: torch.dtype = torch.float32
) -> DataLoader:
    encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            pretrained,
            subfolder = 'text_encoder'
    )
    encoder = encoder \
            .cpu() \
            .to(dtype = torch.float32, memory_format = torch.contiguous_format) \
            .eval().requires_grad_(False)
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            pretrained,
            subfolder = 'scheduler'
    )
    dataset = datasets.load_dataset(dataset_path, streaming = True)
    merged_dataset = ChainDataset([ dataset[s] for s in dataset ])
    dataloader = DataLoader(
        merged_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        persistent_workers = num_workers > 0,
        drop_last = True,
        collate_fn = partial(collate_fn,
                encoder = encoder,
                noise_scheduler = noise_scheduler,
                num_frames = num_frames,
                hint_spacing = hint_spacing,
                dtype = dtype
        )
    )
    return dataloader

