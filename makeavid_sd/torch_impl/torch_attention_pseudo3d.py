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

import torch
from torch import nn

from einops import rearrange

from diffusers.models.attention_processor import Attention
#from torch_cross_attention import CrossAttention


class TransformerPseudo3DModelOutput:
    def __init__(self, sample: torch.FloatTensor) -> None:
        self.sample = sample


class TransformerPseudo3DModel(nn.Module):
    def __init__(self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 8,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        # its continuous

        # 2. Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
                num_groups = 32,
                num_channels = in_channels,
                eps = 1e-5,
                affine = True
        )
        self.proj_in = nn.Conv2d(
                in_channels,
                inner_dim,
                kernel_size = 1,
                stride = 1,
                padding = 0
        )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockPseudo3D(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout = dropout,
                        cross_attention_dim = cross_attention_dim,
                        attention_bias = attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> TransformerPseudo3DModelOutput:
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, context dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.attention.Transformer2DModelOutput`] or `tuple`: [`~models.attention.Transformer2DModelOutput`]
            if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample
            tensor.
        """
        is_video = hidden_states.ndim == 5
        w = hidden_states.shape[-1]
        f = None
        if is_video:
            b, c, f, h, w = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')

        # 1. Input
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                    hidden_states,
                    context = encoder_hidden_states,
                    num_frames = f
            )

        # 3. Output
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', w = w)
        hidden_states = self.proj_out(hidden_states)
        output = hidden_states + residual

        if is_video:
            output = rearrange(output, '(b f) c h w -> b c f h w', b = b)

        return TransformerPseudo3DModelOutput(sample = output)



class BasicTransformerBlockPseudo3D(nn.Module):
    def __init__(self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout: float = 0.0,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
                query_dim = dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
                query_dim = dim,
                cross_attention_dim = cross_attention_dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # self-attention if context is none
        # temporal self attention
        self.norm_temporal = nn.LayerNorm(dim)
        self.attn_temporal = Attention(
                query_dim = dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # self-attention
        # TODO try temporal cross attention
        """
        self.norm_temporal_cross = nn.LayerNorm(dim)
        self.attn_temporal_cross = Attention(
                query_dim = dim,
                cross_attention_dim = cross_attention_dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # self-attention
        """
        # feed forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout = dropout)

    def forward(self,
            hidden_states: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            num_frames: Optional[int] = None
    ) -> torch.Tensor:
        if context is not None and num_frames is not None:
            context = context.repeat_interleave(num_frames, 0)
        # Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(
                norm_hidden_states
        ) + hidden_states
        # Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(
                norm_hidden_states,
                encoder_hidden_states = context
        ) + hidden_states
        """
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
        """
        # temporal attention
        if num_frames is not None:
            bf, s, c = hidden_states.shape
            hidden_states = rearrange(
                    hidden_states,
                    '(b f) s c -> (b s) f c',
                    f = num_frames
            )
            # temporal self attention
            norm_hidden_states = self.norm_temporal(hidden_states)
            hidden_states = self.attn_temporal(
                    norm_hidden_states
            ) + hidden_states
            # TODO try temporal cross attention
            """
            norm_hidden_states = self.norm_temporal_cross(hidden_states)
            hidden_states= self.attn_temporal_cross(
                    norm_hidden_states,
                    encoder_hidden_states = context
            ) + hidden_states
            """
            hidden_states = rearrange(
                    hidden_states,
                    '(b s) f c -> (b f) s c',
                    f = num_frames,
                    s = s
            )
        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net(hidden_states)
        return hidden_states


# https://arxiv.org/abs/2002.05202
class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim = -1)
        return hidden_states * torch.nn.functional.gelu(gate)

