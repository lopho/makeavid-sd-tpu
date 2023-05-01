
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .flax_attention_pseudo3d import TransformerPseudo3DModel
from .flax_resnet_pseudo3d import ResnetBlockPseudo3D, DownsamplePseudo3D, UpsamplePseudo3D


class UNetMidBlockPseudo3DCrossAttn(nn.Module):
    in_channels: int
    num_layers: int = 1
    attn_num_head_channels: int = 1
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        resnets = [
                ResnetBlockPseudo3D(
                        in_channels = self.in_channels,
                        out_channels = self.in_channels,
                        dtype = self.dtype
                )
        ]
        attentions = []
        for _ in range(self.num_layers):
            attn_block = TransformerPseudo3DModel(
                    in_channels = self.in_channels,
                    num_attention_heads = self.attn_num_head_channels,
                    attention_head_dim = self.in_channels // self.attn_num_head_channels,
                    num_layers = 1,
                    use_memory_efficient_attention = self.use_memory_efficient_attention,
                    dtype = self.dtype
            )
            attentions.append(attn_block)
            res_block = ResnetBlockPseudo3D(
                    in_channels = self.in_channels,
                    out_channels = self.in_channels,
                    dtype = self.dtype
            )
            resnets.append(res_block)
        self.attentions = attentions
        self.resnets = resnets

    def __call__(self,
            hidden_states: jax.Array,
            temb: jax.Array,
            encoder_hidden_states = jax.Array
    ) -> jax.Array:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class CrossAttnDownBlockPseudo3D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_downsample: bool = True
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        attentions = []
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = ResnetBlockPseudo3D(
                    in_channels = in_channels,
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
            resnets.append(res_block)
            attn_block = TransformerPseudo3DModel(
                    in_channels = self.out_channels,
                    num_attention_heads = self.attn_num_head_channels,
                    attention_head_dim = self.out_channels // self.attn_num_head_channels,
                    num_layers = 1,
                    use_memory_efficient_attention = self.use_memory_efficient_attention,
                    dtype = self.dtype
            )
            attentions.append(attn_block)
        self.resnets = resnets
        self.attentions = attentions

        if self.add_downsample:
            self.downsamplers_0 = DownsamplePseudo3D(
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
        else:
            self.downsamplers_0 = None

    def __call__(self,
            hidden_states: jax.Array,
            temb: jax.Array,
            encoder_hidden_states: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states += (hidden_states, )
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states, )
        return hidden_states, output_states


class DownBlockPseudo3D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = ResnetBlockPseudo3D(
                    in_channels = in_channels,
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
            resnets.append(res_block)
        self.resnets = resnets
        if self.add_downsample:
            self.downsamplers_0 = DownsamplePseudo3D(
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
        else:
            self.downsamplers_0 = None

    def __call__(self,
            hidden_states: jax.Array,
            temb: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states, )
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states, )
        return hidden_states, output_states


class CrossAttnUpBlockPseudo3D(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channels: int
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_upsample: bool = True
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        resnets = []
        attentions = []
        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers -1) else self.out_channels
            resnet_in_channels = self.prev_output_channels if i == 0 else self.out_channels
            res_block = ResnetBlockPseudo3D(
                    in_channels = resnet_in_channels + res_skip_channels,
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
            resnets.append(res_block)
            attn_block = TransformerPseudo3DModel(
                    in_channels = self.out_channels,
                    num_attention_heads = self.attn_num_head_channels,
                    attention_head_dim = self.out_channels // self.attn_num_head_channels,
                    num_layers = 1,
                    use_memory_efficient_attention = self.use_memory_efficient_attention,
                    dtype = self.dtype
            )
            attentions.append(attn_block)
        self.resnets = resnets
        self.attentions = attentions
        if self.add_upsample:
            self.upsamplers_0 = UpsamplePseudo3D(
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
        else:
            self.upsamplers_0 = None

    def __call__(self,
            hidden_states: jax.Array,
            res_hidden_states_tuple: Tuple[jax.Array, ...],
            temb: jax.Array,
            encoder_hidden_states: jax.Array
    ) -> jax.Array:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis = -1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class UpBlockPseudo3D(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channels: int
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        resnets = []
        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channels if i == 0 else self.out_channels
            res_block = ResnetBlockPseudo3D(
                    in_channels = resnet_in_channels + res_skip_channels,
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
            resnets.append(res_block)
        self.resnets = resnets
        if self.add_upsample:
            self.upsamplers_0 = UpsamplePseudo3D(
                    out_channels = self.out_channels,
                    dtype = self.dtype
            )
        else:
            self.upsamplers_0 = None

    def __call__(self,
            hidden_states: jax.Array,
            res_hidden_states_tuple: Tuple[jax.Array, ...],
            temb: jax.Array
    ) -> jax.Array:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate([hidden_states, res_hidden_states], axis = -1)
            hidden_states = resnet(hidden_states, temb)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states

