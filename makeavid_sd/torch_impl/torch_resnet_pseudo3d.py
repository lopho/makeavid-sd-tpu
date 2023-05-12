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

from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvPseudo3D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Tuple[int, int], int],
            stride: Union[Tuple[int, int], int] = 1,
            padding: Union[Tuple[int, int], int, str] = 'same',
            legacy_v010: bool = True
    ) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
        )
        self.temporal_conv = nn.Conv1d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 3 if legacy_v010 else kernel_size,
                padding = 1 if legacy_v010 else 'same'
        )
        # dirac impulse |= conv identity
        nn.init.dirac_(self.temporal_conv.weight.data)
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_video = x.ndim == 5
        if is_video:
            b, c, f, h, w = x.shape
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.spatial_conv(x)
        if not is_video:
            return x
        else:
            bf, c, h, w = x.shape
            x = rearrange(x, '(b f) c h w -> (b h w) c f', b = b, f = f)
            x = self.temporal_conv(x)
            x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)
            return x


class UpsamplePseudo3D(nn.Module):
    def __init__(self,
                channels: int,
                out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = ConvPseudo3D(
                in_channels = self.channels,
                out_channels = self.out_channels,
                kernel_size = 3,
                padding = 1
        )

    def forward(self, hidden_states, upsample_size = None):
        # TODO remove once bfloat interpolate on cuda is implemented in torch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
        is_video = hidden_states.ndim == 5
        if is_video:
            b, *_ = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if upsample_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor = 2.0, mode = 'nearest')
        else:
            hidden_states = F.interpolate(hidden_states, size = upsample_size, mode = 'nearest')
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        if is_video:
            hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w', b = b)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class DownsamplePseudo3D(nn.Module):
    def __init__(self,
            channels: int,
            out_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = ConvPseudo3D(self.channels, self.out_channels, 3, stride = 2, padding = 1)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlockPseudo3D(nn.Module):
    def __init__(self,
            *,
            in_channels: int,
            out_channels: Optional[int] = None,
            temb_channels: int = 512,
            dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = torch.nn.GroupNorm(
                num_groups = 32,
                num_channels = in_channels,
                eps = 1e-5,
                affine = True
        )
        self.conv1 = ConvPseudo3D(
                in_channels,
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
        )
        self.time_emb_proj = torch.nn.Linear(
                temb_channels,
                out_channels
        )
        self.norm2 = torch.nn.GroupNorm(
                num_groups = 32,
                num_channels = out_channels,
                eps = 1e-5,
                affine = True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ConvPseudo3D(
                out_channels,
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = ConvPseudo3D(
                in_channels,
                out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
        ) if in_channels != out_channels else None

    def forward(self,
            input_tensor: torch.Tensor,
            temb: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        is_video = hidden_states.ndim == 5
        if is_video:
            b, c, f, h, w = hidden_states.shape
            hidden_states = rearrange(
                    hidden_states,
                    'b c f h w -> (b f) c h w'
            )
            hidden_states = hidden_states + temb.repeat_interleave(f, 0)
            hidden_states = rearrange(
                    hidden_states,
                    '(b f) c h w -> b c f h w',
                    b = b
            )
        else:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = input_tensor + hidden_states
        return output_tensor

