# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from nemo.collections.tts.modules.transformer_2501 import PositionwiseConvFF, CrossAttention, Transformer


class PerceiverLayer(torch.nn.Module):
    def __init__(
        self,
        num_latents: int,
        d_model: int,
        d_ffn: int,
        kernel_size: int,
        p_dropout: float,
        xa_d_memory: int,
        xa_n_heads: int,
        apply_norm_to_cond: bool = True,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        """
        One layer of the Percevier(https://arxiv.org/abs/2103.03206) model.
        Args:
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            xa_d_memory <int>: Hidden dimension for cross attention
            xa_n_heads <int>: Number of attention heads used in cross attention
            apply_norm_to_cond <bool>: Whether to apply normalization to input tensor (memory of xattn)
            conv_non_linearity <Callable>: Convolution non-linearity
        """
        super().__init__()

        self.norm_self = torch.nn.LayerNorm(d_model, bias=False)
        self.cross_attention = CrossAttention(
            n_heads=xa_n_heads,
            d_model=d_model,
            d_memory=xa_d_memory,
            p_dropout=p_dropout,
        )

        self.norm_xattn_memory = torch.nn.Identity()
        if apply_norm_to_cond:
            self.norm_xattn_memory = torch.nn.LayerNorm(xa_d_memory, bias=False)

        self.norm_pos_ff = torch.nn.LayerNorm(d_model, bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, d_ffn, p_dropout, kernel_size=kernel_size, non_linearity=conv_non_linearity, is_causal=False
        )

    def forward(
        self,
        latents: torch.Tensor,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> Dict:
        """
        Args:
            x <torch tensor> (B, T1, C): Input tensor
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mask for self attention input.

        Returns dict with keys
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <dict>: Attention probabilities
        """
        cond = cond * cond_mask.unsqueeze(-1)
        cond_normed = self.norm_xattn_memory(cond)

        latents_res, latents_attn_prob = self.cross_attention(
            query=latents, query_mask=None, memory=cond_normed, memory_mask=cond_mask
        )
        latents = latents + latents_res
        latents = latents + self.pos_ff(self.norm_pos_ff(latents))

        return {
            'output': latents,
            'attn_probabilities': {'self_attn_probabilities': None, 'cross_attn_probabilities': latents_attn_prob},
        }


class Perceiver(torch.nn.Module):
    def __init__(
        self,
        num_latents,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        kernel_size: int,
        p_dropout: float,
        xa_d_memory: int,
        xa_n_heads: int,
        p_dropout_out: float = 0.0,
        apply_norm_to_cond: bool = True,
        apply_norm_out: bool = True,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p_dropout)

        self.dropout_out = torch.nn.Identity()
        if p_dropout_out > 0.0:
            self.dropout_out = torch.nn.Dropout(p_dropout_out)

        self.norm_out = torch.nn.Identity()
        if apply_norm_out:
            self.norm_out = torch.nn.LayerNorm(d_model, bias=False)

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                PerceiverLayer(
                    num_latents=num_latents,
                    d_model=d_model,
                    d_ffn=d_ffn,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    xa_d_memory=xa_d_memory,
                    xa_n_heads=xa_n_heads,
                    apply_norm_to_cond=apply_norm_to_cond,
                    conv_non_linearity=conv_non_linearity,
                )
            )

        self.latents = torch.nn.Parameter(torch.randn(num_latents, d_model))
        torch.nn.init.normal_(self.latents, std=0.02)
        self.apply(Transformer._init_weights_gpt2)
        for name, param in self.named_parameters():
            if 'o_net' in name and name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Args:
            x <torch tensor> (B, T1, C):
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mostly used in non-causal self-attention to zero out padding values. In causal self-attention, the
                causal mask will be used in place of this.
            cond <torch tensor> (B, T2, C) or list of such tensors (from different encoders)
            cond_mask <bool mask> (B, T2): Multiplicative mask where True means we keep the input, False we zero it
                out or list of such tensors (from different encoders) output <torch tensor> (B, T1, C)
            multi_encoder_mapping <list> <int>: None or Same size as n_layers, value indicates which cond input to use
                for this layer

        Returns dict with keys:
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <list>: Attention probabilities of each layer
        """

        attn_probabilities = []
        x = self.dropout(x)
        latents = torch.repeat_interleave(self.latents.unsqueeze(0), x.shape[0], dim=0)
        for layer in self.layers:
            out_dict = layer(latents=latents, cond=x, cond_mask=x_mask)
            latents = out_dict['output']
            attn_probabilities.append(out_dict['attn_probabilities'])

        latents = self.norm_out(latents)
        latents = self.dropout_out(latents)

        return {'output': latents, 'attn_probabilities': attn_probabilities}
