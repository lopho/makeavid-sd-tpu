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

from typing import Tuple, Union

import torch
from torch import nn
import torch.nn as nn

#from .torch_embeddings import TimestepEmbedding, Timesteps
from .torch_unet_pseudo3d_blocks import (
    UNetMidBlockPseudo3DCrossAttn,
    DownBlockPseudo3D,
    CrossAttnDownBlockPseudo3D,
    UpBlockPseudo3D,
    CrossAttnUpBlockPseudo3D
)
from .torch_resnet_pseudo3d import ConvPseudo3D
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin


@dataclass
class UNetPseudo3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNetPseudo3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
            in_channels: int = 9,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlockPseudo3D",
                    "CrossAttnDownBlockPseudo3D",
                    "CrossAttnDownBlockPseudo3D",
                    "DownBlockPseudo3D",
            ),
            up_block_types: Tuple[str] = (
                    "UpBlockPseudo3D",
                    "CrossAttnUpBlockPseudo3D",
                    "CrossAttnUpBlockPseudo3D",
                    "CrossAttnUpBlockPseudo3D"
            ),
            block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int, ...]] = 2,
            cross_attention_dim: Union[int, Tuple[int, ...]] = 768,
            attention_head_dim: Union[int, Tuple[int, ...]] = 8,
            **kwargs
    ) -> None:
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = (layers_per_block,) * len(down_block_types)

        # input
        self.conv_in = ConvPseudo3D(
                in_channels,
                block_out_channels[0],
                kernel_size = 3,
                padding = (1, 1)
        )

        # time
        self.time_proj = Timesteps(
                block_out_channels[0],
                flip_sin_to_cos,
                freq_shift
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim
        )

        self.down_blocks = nn.ModuleList()
        self.mid_block = None
        self.up_blocks = nn.ModuleList()

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type in ['DownBlock2D', 'DownBlockPseudo3D']:
                down_block = DownBlockPseudo3D(
                        num_layers = layers_per_block[i],
                        in_channels = input_channel,
                        out_channels = output_channel,
                        temb_channels = time_embed_dim,
                        add_downsample = not is_final_block
                )
            elif down_block_type in ['CrossAttnDownBlock2D', 'CrossAttnDownBlockPseudo3D']:
                down_block = CrossAttnDownBlockPseudo3D(
                        num_layers = layers_per_block[i],
                        in_channels = input_channel,
                        out_channels = output_channel,
                        temb_channels = time_embed_dim,
                        add_downsample = not is_final_block,
                        cross_attention_dim = cross_attention_dim[i],
                        attn_num_head_channels = attention_head_dim[i]
                )
            else:
                raise NotImplementedError(down_block_type)
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockPseudo3DCrossAttn(
                in_channels = block_out_channels[-1],
                temb_channels = time_embed_dim,
                cross_attention_dim = cross_attention_dim[-1],
                attn_num_head_channels = attention_head_dim[-1]
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_layers_per_block = list(reversed(layers_per_block))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            if up_block_type in ['UpBlock2D', 'UpBlockPseudo3D']:
                up_block = UpBlockPseudo3D(
                        num_layers = reversed_layers_per_block[i] + 1,
                        in_channels = input_channel,
                        out_channels = output_channel,
                        prev_output_channel = prev_output_channel,
                        temb_channels = time_embed_dim,
                        add_upsample = add_upsample
                )
            elif up_block_type in ['CrossAttnUpBlock2D', 'CrossAttnUpBlockPseudo3D']:
                up_block = CrossAttnUpBlockPseudo3D(
                        num_layers = reversed_layers_per_block[i] + 1,
                        in_channels = input_channel,
                        out_channels = output_channel,
                        prev_output_channel = prev_output_channel,
                        temb_channels = time_embed_dim,
                        add_upsample = add_upsample,
                        cross_attention_dim = reversed_cross_attention_dim[i],
                        attn_num_head_channels = reversed_attention_head_dim[i]
                )
            else:
                raise NotImplementedError(up_block_type)

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
                num_channels = block_out_channels[0],
                num_groups = 32,
                eps = 1e-5
        )
        self.conv_act = nn.SiLU()
        self.conv_out = ConvPseudo3D(
                block_out_channels[0],
                out_channels,
                3,
                padding = 1
        )


    def forward(
        self,
        sample: torch.FloatTensor,
        timesteps: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor
    ) -> Union[UNetPseudo3DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype = sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'attentions') and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states = sample,
                    temb = emb,
                    encoder_hidden_states = encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, 'attentions') and upsample_block.attentions is not None:
                sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        encoder_hidden_states = encoder_hidden_states,
                        upsample_size = upsample_size
                )
            else:
                sample = upsample_block(
                        hidden_states = sample,
                        temb = emb,
                        res_hidden_states_tuple = res_samples,
                        upsample_size = upsample_size
                )
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return UNetPseudo3DConditionOutput(sample = sample)
