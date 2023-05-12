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

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

import einops

from diffusers.models.attention_flax import FlaxAttention


class TransformerPseudo3DModel(nn.Module):
    in_channels: int
    num_attention_heads: int
    attention_head_dim: int
    num_layers: int = 1
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        inner_dim = self.num_attention_heads * self.attention_head_dim
        self.norm = nn.GroupNorm(
                num_groups = 32,
                epsilon = 1e-5
        )
        self.proj_in = nn.Conv(
                inner_dim,
                kernel_size = (1, 1),
                strides = (1, 1),
                padding = 'VALID',
                dtype = self.dtype
        )
        transformer_blocks = []
        #CheckpointTransformerBlock = nn.checkpoint(
        #        BasicTransformerBlockPseudo3D,
        #        static_argnums = (2,3,4)
        #        #prevent_cse = False
        #)
        CheckpointTransformerBlock = BasicTransformerBlockPseudo3D
        for _ in range(self.num_layers):
            transformer_blocks.append(CheckpointTransformerBlock(
                        dim = inner_dim,
                        num_attention_heads = self.num_attention_heads,
                        attention_head_dim = self.attention_head_dim,
                        use_memory_efficient_attention = self.use_memory_efficient_attention,
                        dtype = self.dtype
                ))
        self.transformer_blocks = transformer_blocks
        self.proj_out = nn.Conv(
                inner_dim,
                kernel_size = (1, 1),
                strides = (1, 1),
                padding = 'VALID',
                dtype = self.dtype
        )

    def __call__(self,
            hidden_states: jax.Array,
            encoder_hidden_states: Optional[jax.Array] = None
    ) -> jax.Array:
        is_video = hidden_states.ndim == 5
        f: Optional[int] = None
        if is_video:
            # jax is channels last
            # b,c,f,h,w WRONG
            # b,f,h,w,c CORRECT
            # b, c, f, h, w = hidden_states.shape
            #hidden_states = einops.rearrange(hidden_states, 'b c f h w -> (b f) c h w')
            b, f, h, w, c = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, 'b f h w c -> (b f) h w c')

        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.reshape(batch, height * width, channels)
        for block in self.transformer_blocks:
            hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    f,
                    height,
                    width
            )
        hidden_states = hidden_states.reshape(batch, height, width, channels)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        if is_video:
            hidden_states = einops.rearrange(hidden_states, '(b f) h w c -> b f h w c', b = b)
        return hidden_states


class BasicTransformerBlockPseudo3D(nn.Module):
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.attn1 = FlaxAttention(
                query_dim = self.dim,
                heads = self.num_attention_heads,
                dim_head = self.attention_head_dim,
                use_memory_efficient_attention = self.use_memory_efficient_attention,
                dtype = self.dtype
        )
        self.ff = FeedForward(dim = self.dim, dtype = self.dtype)
        self.attn2 = FlaxAttention(
                query_dim = self.dim,
                heads = self.num_attention_heads,
                dim_head = self.attention_head_dim,
                use_memory_efficient_attention = self.use_memory_efficient_attention,
                dtype = self.dtype
        )
        self.attn_temporal = FlaxAttention(
                query_dim = self.dim,
                heads = self.num_attention_heads,
                dim_head = self.attention_head_dim,
                use_memory_efficient_attention = self.use_memory_efficient_attention,
                dtype = self.dtype
        )
        self.norm1 = nn.LayerNorm(epsilon = 1e-5, dtype = self.dtype)
        self.norm2 = nn.LayerNorm(epsilon = 1e-5, dtype = self.dtype)
        self.norm_temporal = nn.LayerNorm(epsilon = 1e-5, dtype = self.dtype)
        self.norm3 = nn.LayerNorm(epsilon = 1e-5, dtype = self.dtype)

    def __call__(self,
            hidden_states: jax.Array,
            context: Optional[jax.Array] = None,
            frames_length: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None
    ) -> jax.Array:
        if context is not None and frames_length is not None:
            context = context.repeat(frames_length, axis = 0)
        # self attention
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(norm_hidden_states) + hidden_states
        # cross attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(
                norm_hidden_states,
                context = context
        ) + hidden_states
        # temporal attention
        if frames_length is not None:
            #bf, hw, c = hidden_states.shape
            # (b f) (h w) c -> b f (h w) c
            #hidden_states = hidden_states.reshape(bf // frames_length, frames_length, hw, c)
            #b, f, hw, c = hidden_states.shape
            # b f (h w) c -> b (h w) f c
            #hidden_states = hidden_states.transpose(0, 2, 1, 3)
            # b (h w) f c -> (b h w) f c
            #hidden_states = hidden_states.reshape(b * hw, frames_length, c)
            hidden_states = einops.rearrange(
                    hidden_states,
                    '(b f) (h w) c -> (b h w) f c',
                    f = frames_length,
                    h = height,
                    w = width
            )
            norm_hidden_states = self.norm_temporal(hidden_states)
            hidden_states = self.attn_temporal(norm_hidden_states) + hidden_states
            # (b h w) f c -> b (h w) f c
            #hidden_states = hidden_states.reshape(b, hw, f, c)
            # b (h w) f c -> b f (h w) c
            #hidden_states = hidden_states.transpose(0, 2, 1, 3)
            # b f h w c -> (b f) (h w) c
            #hidden_states = hidden_states.reshape(bf, hw, c)
            hidden_states = einops.rearrange(
                    hidden_states,
                    '(b h w) f c -> (b f) (h w) c',
                    f = frames_length,
                    h = height,
                    w = width
            )
        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # _0/_2 naming for compatibility with torch.nn.ModuleList
        self.net_0 = GEGLU(self.dim, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype = self.dtype)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype = self.dtype)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis = 2)
        return hidden_linear * nn.gelu(hidden_gelu)

