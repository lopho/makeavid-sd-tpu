
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from diffusers.utils import BaseOutput

from flax_unet_pseudo3d_blocks import (
        CrossAttnDownBlockPseudo3D,
        CrossAttnUpBlockPseudo3D,
        DownBlockPseudo3D,
        UpBlockPseudo3D,
        UNetMidBlockPseudo3DCrossAttn
)
from flax_embeddings import (
        TimestepEmbedding,
        Timesteps
)
from flax_resnet_pseudo3d import ConvPseudo3D


class UNetPseudo3DConditionOutput(BaseOutput):
    sample: jax.Array


@flax_register_to_config
class UNetPseudo3DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    sample_size: Tuple[int, int] = (64, 64)
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str] = (
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "DownBlockPseudo3D"
    )
    up_block_types: Tuple[str] = (
            "UpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D",
            "CrossAttnUpBlockPseudo3D"
    )
    block_out_channels: Tuple[int] = (
            320,
            640,
            1280,
            1280
    )
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int]] = 8
    cross_attention_dim: int = 768
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def init_weights(self, rng: jax.random.KeyArray) -> FrozenDict:
        sample_shape = (1, self.in_channels, 1, *self.sample_size)
        sample = jnp.zeros(sample_shape, dtype = self.dtype)
        timesteps = jnp.ones((1, ), dtype = jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype = self.dtype)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = { "params": params_rng, "dropout": dropout_rng }
        return self.init(rngs, sample, timesteps, encoder_hidden_states)["params"]

    def setup(self) -> None:
        if isinstance(self.attention_head_dim, int):
            attention_head_dim = (self.attention_head_dim, ) * len(self.down_block_types)
        else:
            attention_head_dim = self.attention_head_dim
        time_embed_dim = self.block_out_channels[0] * 4
        self.conv_in = ConvPseudo3D(
                features = self.block_out_channels[0],
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )
        self.time_proj = Timesteps(
                dim = self.block_out_channels[0],
                flip_sin_to_cos = self.flip_sin_to_cos,
                freq_shift = self.freq_shift,
                dtype = self.dtype
        )
        self.time_embedding = TimestepEmbedding(
                time_embed_dim = time_embed_dim,
                dtype = self.dtype
        )
        down_blocks = []
        output_channels = self.block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channels = output_channels
            output_channels = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1
            if down_block_type == 'CrossAttnDownBlockPseudo3D':
                down_block = CrossAttnDownBlockPseudo3D(
                        in_channels = input_channels,
                        out_channels = output_channels,
                        num_layers = self.layers_per_block,
                        attn_num_head_channels = attention_head_dim[i],
                        add_downsample = not is_final_block,
                        use_memory_efficient_attention = self.use_memory_efficient_attention,
                        dtype = self.dtype
                )
            elif down_block_type == 'DownBlockPseudo3D':
                down_block = DownBlockPseudo3D(
                        in_channels = input_channels,
                        out_channels = output_channels,
                        num_layers = self.layers_per_block,
                        add_downsample = not is_final_block,
                        dtype = self.dtype
                )
            else:
                raise NotImplementedError(f'Unimplemented down block type: {down_block_type}')
            down_blocks.append(down_block)
        self.down_blocks = down_blocks
        self.mid_block = UNetMidBlockPseudo3DCrossAttn(
                in_channels = self.block_out_channels[-1],
                attn_num_head_channels = attention_head_dim[-1],
                use_memory_efficient_attention = self.use_memory_efficient_attention,
                dtype = self.dtype
        )
        up_blocks = []
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        output_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channels = output_channels
            output_channels = reversed_block_out_channels[i]
            input_channels = reversed_block_out_channels[min(i + 1, len(self.block_out_channels) - 1)]
            is_final_block = i == len(self.block_out_channels) - 1
            if up_block_type == 'CrossAttnUpBlockPseudo3D':
                up_block = CrossAttnUpBlockPseudo3D(
                        in_channels = input_channels,
                        out_channels = output_channels,
                        prev_output_channels = prev_output_channels,
                        num_layers = self.layers_per_block + 1,
                        attn_num_head_channels = reversed_attention_head_dim[i],
                        add_upsample = not is_final_block,
                        use_memory_efficient_attention = self.use_memory_efficient_attention,
                        dtype = self.dtype
                )
            elif up_block_type == 'UpBlockPseudo3D':
                up_block = UpBlockPseudo3D(
                        in_channels = input_channels,
                        out_channels = output_channels,
                        prev_output_channels = prev_output_channels,
                        num_layers = self.layers_per_block + 1,
                        add_upsample = not is_final_block,
                        dtype = self.dtype
                )
            else:
                raise NotImplementedError(f'Unimplemented up block type: {up_block_type}')
            up_blocks.append(up_block)
        self.up_blocks = up_blocks
        self.conv_norm_out = nn.GroupNorm(
                num_groups = 32,
                epsilon = 1e-5,
                dtype = self.dtype
        )
        self.conv_out = ConvPseudo3D(
                features = self.out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = ((1, 1), (1, 1)),
                dtype = self.dtype
        )

    def __call__(self,
            sample: jax.Array,
            timesteps: jax.Array,
            encoder_hidden_states: jax.Array,
            return_dict: bool = True
    ) -> Union[UNetPseudo3DConditionOutput, Tuple[jax.Array]]:
        if timesteps.dtype != self.dtype:
            timesteps = timesteps.astype(dtype = self.dtype)
        if timesteps.ndim == 0:
            timesteps = jnp.expand_dims(timesteps, 0)
        # b,c,f,h,w -> b,f,h,w,c
        sample = jnp.transpose(sample, (0, 2, 3, 4, 1))

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        sample = self.conv_in(sample)
        down_block_res_samples = (sample, )
        for down_block in self.down_blocks:
            if isinstance(down_block, CrossAttnDownBlockPseudo3D):
                sample, res_samples = down_block(
                        hidden_states = sample,
                        temb = t_emb,
                        encoder_hidden_states = encoder_hidden_states
                )
            elif isinstance(down_block, DownBlockPseudo3D):
                sample, res_samples = down_block(
                        hidden_states = sample,
                        temb = t_emb
                )
            else:
                raise NotImplementedError(f'Unimplemented down block type: {down_block.__class__.__name__}')
            down_block_res_samples += res_samples
        sample = self.mid_block(
                hidden_states = sample,
                temb = t_emb,
                encoder_hidden_states = encoder_hidden_states
        )
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.layers_per_block + 1):]
            down_block_res_samples = down_block_res_samples[:-(self.layers_per_block + 1)]
            if isinstance(up_block, CrossAttnUpBlockPseudo3D):
                sample = up_block(
                        hidden_states = sample,
                        temb = t_emb,
                        encoder_hidden_states = encoder_hidden_states,
                        res_hidden_states_tuple = res_samples
                )
            elif isinstance(up_block, UpBlockPseudo3D):
                sample = up_block(
                        hidden_states = sample,
                        temb = t_emb,
                        res_hidden_states_tuple = res_samples
                )
            else:
                raise NotImplementedError(f'Unimplemented up block type: {up_block.__class__.__name__}')
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        # b,f,h,w,c -> b,c,f,h,w
        sample = jnp.transpose(sample, (0, 4, 1, 2, 3))
        if not return_dict:
            return (sample, )
        return UNetPseudo3DConditionOutput(sample = sample)

