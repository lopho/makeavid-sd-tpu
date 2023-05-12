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

import os
os.environ['PJRT_DEVICE'] = 'TPU'

from tqdm.auto import tqdm

from torch_xla.core import xla_model
import torch
from torch.utils.data import DataLoader
from ..torch_impl.torch_unet_pseudo3d_condition import UNetPseudo3DConditionModel

class TrainerUNetPseudo3D:
    def __init__(self,
            pretrained: str = 'lxj616/make-a-stable-diffusion-video-timelapse',
            lr: float = 1e-4,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dtype = dtype
        self.device: torch.device = xla_model.xla_device(0)
        unet: UNetPseudo3DConditionModel = UNetPseudo3DConditionModel.from_pretrained(
                pretrained,
                subfolder = 'unet'
        ).to(dtype = dtype, memory_format = torch.contiguous_format)
        unfreeze_all: bool = False
        unet = unet.train()
        if not unfreeze_all:
            unet.requires_grad_(False)
            for name, param in unet.named_parameters():
                if 'temporal_conv' in name:
                    param.requires_grad_(True)
            for block in [*unet.down_blocks, unet.mid_block, *unet.up_blocks]:
                if hasattr(block, 'attentions') and block.attentions is not None:
                    for attn_block in block.attentions:
                        for transformer_block in attn_block.transformer_blocks:
                            transformer_block.requires_grad_(False)
                            transformer_block.attn_temporal.requires_grad_(True)
                            transformer_block.norm_temporal.requires_grad_(True)
        else:
            unet.requires_grad_(True)
        self.model: UNetPseudo3DConditionModel = unet.to(device = self.device)
        #self.model = torch.compile(self.model, backend = 'aot_torchxla_trace_once')
        self.params = lambda: filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim: torch.optim.Optimizer = torch.optim.AdamW(self.params(), lr = lr)
        def lr_warmup(warmup_steps: int = 0):
            def lambda_lr(step: int) -> float:
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return 1.0
            return lambda_lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda = lr_warmup(warmup_steps = 60), last_epoch = -1)

    @torch.no_grad()
    def train(self, dataloader: DataLoader, epochs: int = 1, log_every: int = 1, save_every: int = 1000) -> None:
        # 'latent_model_input'
        # 'encoder_hidden_states'
        # 'timesteps'
        # 'noise'
        global_step: int = 0
        for epoch in range(epochs):
            pbar = tqdm(dataloader, dynamic_ncols = True, smoothing = 0.01)
            for b in pbar:
                latent_model_input: torch.Tensor = b['latent_model_input'].to(device = self.device)
                encoder_hidden_states: torch.Tensor = b['encoder_hidden_states'].to(device = self.device)
                timesteps: torch.Tensor = b['timesteps'].to(device = self.device)
                noise: torch.Tensor = b['noise'].to(device = self.device)
                with torch.enable_grad():
                    self.optim.zero_grad(set_to_none = True)
                    y = self.model(latent_model_input, timesteps, encoder_hidden_states).sample
                    loss = torch.nn.functional.mse_loss(noise, y)
                    loss.backward()
                    self.optim.step()
                    self.scheduler.step()
                    xla_model.mark_step()
                if global_step % log_every == 0:
                    pbar.set_postfix({ 'loss': loss.detach().item(), 'epoch': epoch })

