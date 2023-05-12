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

from typing import Optional, Union, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

import einops


class ConvPseudo3D(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: nn.linear.PaddingLike = 'SAME'
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.spatial_conv = nn.Conv(
                features = self.features,
                kernel_size = self.kernel_size,
                strides = self.strides,
                padding = self.padding,
                dtype = self.dtype
        )
        self.temporal_conv = nn.Conv(
                features = self.features,
                kernel_size = (3,),
                padding = 'SAME',
                dtype = self.dtype,
                bias_init = nn.initializers.zeros_init()
                # TODO dirac delta (identity) initialization impl
                # kernel_init = torch.nn.init.dirac_ <-> jax/lax
        )

    def __call__(self, x: jax.Array, convolve_across_time: bool = True) -> jax.Array:
        is_video = x.ndim == 5
        convolve_across_time = convolve_across_time and is_video
        if is_video:
            b, f, h, w, c = x.shape
            x = einops.rearrange(x, 'b f h w c -> (b f) h w c')
        x = self.spatial_conv(x)
        if is_video:
            x = einops.rearrange(x, '(b f) h w c -> b f h w c', b = b)
        if not convolve_across_time:
            return x
        if is_video:
            b, f, h, w, c = x.shape
            x = einops.rearrange(x, 'b f h w c -> (b h w) f c')
            x = self.temporal_conv(x)
            x = einops.rearrange(x, '(b h w) f c -> b f h w c', h = h, w = w)
        return x


class UpsamplePseudo3D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.conv = ConvPseudo3D(
                features = self.out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        is_video = hidden_states.ndim == 5
        if is_video:
            b, *_ = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, 'b f h w c -> (b f) h w c')
        batch, h, w, c = hidden_states.shape
        hidden_states = jax.image.resize(
                image = hidden_states,
                shape = (batch, h * 2, w * 2, c),
                method = 'nearest'
        )
        if is_video:
            hidden_states = einops.rearrange(hidden_states, '(b f) h w c -> b f h w c', b = b)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class DownsamplePseudo3D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.conv = ConvPseudo3D(
                features = self.out_channels,
                kernel_size = (3, 3),
                strides = (2, 2),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlockPseudo3D(nn.Module):
    in_channels: int
    out_channels: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.norm1 = nn.GroupNorm(
                num_groups = 32,
                epsilon = 1e-5
        )
        self.conv1 = ConvPseudo3D(
                features = out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )
        self.time_emb_proj = nn.Dense(
                out_channels,
                dtype = self.dtype
        )
        self.norm2 = nn.GroupNorm(
                num_groups = 32,
                epsilon = 1e-5
        )
        self.conv2 = ConvPseudo3D(
                features = out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )
        self.conv_shortcut = ConvPseudo3D(
                features = self.out_channels,
                kernel_size = (1, 1),
                strides = (1, 1),
                padding = 'VALID',
                dtype = self.dtype
        ) if self.in_channels != out_channels else None

    def __call__(self,
            hidden_states: jax.Array,
            temb: jax.Array
    ) -> jax.Array:
        is_video = hidden_states.ndim == 5
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        temb = nn.silu(temb)
        temb = self.time_emb_proj(temb)
        temb = jnp.expand_dims(temb, 1)
        temb = jnp.expand_dims(temb, 1)
        if is_video:
            b, f, *_ = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, 'b f h w c -> (b f) h w c')
            hidden_states = hidden_states + temb.repeat(f, 0)
            hidden_states = einops.rearrange(hidden_states, '(b f) h w c -> b f h w c', b = b)
        else:
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        hidden_states = hidden_states + residual
        return hidden_states

