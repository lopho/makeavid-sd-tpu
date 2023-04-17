from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange


class TransformerPseudo3DModelOutput:
    def __init__(self, sample: torch.FloatTensor) -> None:
        self.sample = sample


class TransformerPseudo3DModel(nn.Module):
    def __init__(self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
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
                num_groups = norm_num_groups,
                num_channels = in_channels,
                eps = 1e-6,
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
                BasicTransformerBlock(
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
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.long] = None
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
        b, c, *_, h, w = hidden_states.shape
        is_video = hidden_states.ndim == 5
        f = None
        if is_video:
            b, c, f, h, w = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')
            #encoder_hidden_states = encoder_hidden_states.repeat_interleave(f, 0)

        # 1. Input
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                    hidden_states,
                    context = encoder_hidden_states,
                    timestep = timestep,
                    frames_length = f,
                    height = height,
                    weight = weight
            )

        # 3. Output
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
        hidden_states = self.proj_out(hidden_states)
        output = hidden_states + residual

        if is_video:
            output = rearrange(output, '(b f) c h w -> b c f h w', b = b)

        return TransformerPseudo3DModelOutput(sample = output)



class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the context vector for cross attention.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout: float = 0.0,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
                query_dim = dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout = dropout)
        self.attn2 = CrossAttention(
                query_dim = dim,
                cross_attention_dim = cross_attention_dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # is self-attn if context is none
        self.attn_temporal = CrossAttention(
                query_dim = dim,
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout = dropout,
                bias = attention_bias
        )  # is a self-attention

        # layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_temporal = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self,
            hidden_states: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            timestep: Optional[torch.long] = None,
            frames_length: Optional[int] = None,
            height: Optional[int] = None,
            weight: Optional[int] = None
    ) -> torch.Tensor:
        # 1. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states)
        )
        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = (
            self.norm2(hidden_states)
        )
        hidden_states = self.attn2(
                norm_hidden_states,
                context = context,
                frames_length = frames_length
        ) + hidden_states

        # append temporal attention
        if frames_length is not None:
            hidden_states = rearrange(
                    hidden_states,
                    '(b f) (h w) c -> (b h w) f c',
                    f = frames_length,
                    h = height,
                    w = weight
            )
            norm_hidden_states = (
                self.norm_temporal(hidden_states)
            )
            hidden_states = self.attn_temporal(norm_hidden_states) + hidden_states
            hidden_states = rearrange(
                    hidden_states,
                    '(b h w) f c -> (b f) (h w) c',
                    f = frames_length,
                    h = height,
                    w = weight
            )

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.n_heads = heads
        self.d_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias = bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias = bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias = bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self,
            hidden_states: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            frames_length: Optional[int] = None
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # attention, what we cannot get enough of
        query = self.to_q(hidden_states)
        has_cond = context is not None
        if context is not None:
            if frames_length is not None:
                context = context.repeat_interleave(frames_length, 0)
            else:
                context = context
        else:
            context = hidden_states
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        if self.flash is not None and not has_cond and self.d_head <= 64:
            hidden_states = self.flash_attention(query, key, value)
        else:
            hidden_states = self.normal_attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim = 2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim = -1)

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, :self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return out

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        half = attn.shape[0] // 2
        attn[half:] = attn[half:].softmax(dim = -1)
        attn[:half] = attn[:half].softmax(dim = -1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return out

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim = -1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
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

        geglu = GEGLU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(geglu)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim = -1)
        return hidden_states * F.gelu(gate)
