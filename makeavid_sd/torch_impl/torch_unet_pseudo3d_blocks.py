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

import torch
from torch import nn

from .torch_attention_pseudo3d import TransformerPseudo3DModel
from .torch_resnet_pseudo3d import DownsamplePseudo3D, ResnetBlockPseudo3D, UpsamplePseudo3D


class UNetMidBlockPseudo3DCrossAttn(nn.Module):
    def __init__(self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            attn_num_head_channels: int = 1,
            cross_attention_dim: int = 1280,
            **kwargs
    ) -> None:
        super().__init__()
        self.attn_num_head_channels = attn_num_head_channels

        # there is always at least one resnet
        resnets = [
            ResnetBlockPseudo3D(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    temb_channels = temb_channels,
                    dropout = dropout
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                TransformerPseudo3DModel(
                        in_channels = in_channels,
                        num_attention_heads = attn_num_head_channels,
                        attention_head_dim = in_channels // attn_num_head_channels,
                        num_layers = 1,
                        cross_attention_dim = cross_attention_dim,
                )
            )
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        temb_channels = temb_channels,
                        dropout = dropout
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb = None, encoder_hidden_states = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states).sample
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlockPseudo3D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            attn_num_head_channels: int = 1,
            cross_attention_dim: int = 1280,
            add_downsample: bool = True
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        temb_channels = temb_channels,
                        dropout = dropout
                )
            )
            attentions.append(
                TransformerPseudo3DModel(
                        in_channels = out_channels,
                        num_attention_heads = attn_num_head_channels,
                        attention_head_dim = out_channels // attn_num_head_channels,
                        num_layers = 1,
                        cross_attention_dim = cross_attention_dim
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsamplePseudo3D(
                            out_channels,
                            out_channels = out_channels
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb = None, encoder_hidden_states = None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states = encoder_hidden_states).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlockPseudo3D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            add_downsample: bool = True
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockPseudo3D(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        temb_channels = temb_channels,
                        dropout = dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsamplePseudo3D(
                        out_channels,
                        out_channels = out_channels
                    )
                ]
            )
        else:
            self.downsamplers = None


    def forward(self, hidden_states, temb = None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlockPseudo3D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            attn_num_head_channels: int = 1,
            cross_attention_dim: int = 1280,
            add_upsample: bool = True
    ) -> None:
        super().__init__()
        resnets = []
        attentions = []
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                    ResnetBlockPseudo3D(
                            in_channels = resnet_in_channels + res_skip_channels,
                            out_channels = out_channels,
                            temb_channels = temb_channels,
                            dropout = dropout
                    )
            )
            attentions.append(
                    TransformerPseudo3DModel(
                            in_channels = out_channels,
                            num_attention_heads = attn_num_head_channels,
                            attention_head_dim = out_channels // attn_num_head_channels,
                            num_layers = 1,
                            cross_attention_dim = cross_attention_dim,
                    )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                    UpsamplePseudo3D(
                            out_channels,
                            out_channels = out_channels
                    )
            ])
        else:
            self.upsamplers = None

    def forward(self,
            hidden_states,
            res_hidden_states_tuple,
            temb = None,
            encoder_hidden_states = None,
            upsample_size = None
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states = encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlockPseudo3D(nn.Module):
    def __init__(self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            add_upsample: bool = True
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                    ResnetBlockPseudo3D(
                            in_channels = resnet_in_channels + res_skip_channels,
                            out_channels = out_channels,
                            temb_channels = temb_channels,
                            dropout = dropout
                    )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                    UpsamplePseudo3D(
                            out_channels,
                            out_channels = out_channels
                    )
            ])
        else:
            self.upsamplers = None


    def forward(self, hidden_states, res_hidden_states_tuple, temb = None, upsample_size = None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

