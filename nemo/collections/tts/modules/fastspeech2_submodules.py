# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is originally from NVIDIA's DeepLearningExamples library:
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/transformer.py

# Edit list:
# - Two different kernel sizes in FFTransformer convolution

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths

# from common.text.symbols import pad_idx, symbols

from nemo.utils import logging


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size[0], 1, (kernel_size[0] // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size[1], 1, (kernel_size[1] // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=-1)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask, -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        return output

    # disabled; slower
    def forward_einsum(self, h, attn_mask=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x n_head x qlen x klen]
        # attn_score = torch.einsum('ibnd,jbnd->bnij', (head_q, head_k))
        attn_score = torch.einsum('bind,bjnd->bnij', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            attn_score.masked_fill_(attn_mask[:, None, None, :], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head]
        #     -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('bnij,bjnd->bind', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        embed_input=True,
        d_embed=None,
        pre_lnorm=False,
    ):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        if embed_input:
            # self.word_emb = nn.Embedding(len(symbols), d_embed or d_model, padding_idx=pad_idx)
            self.word_emb = nn.Embedding(84, d_embed or d_model, padding_idx=83)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
                )
            )

    def forward(self, dec_inp, seq_lens=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = get_mask_from_lengths(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != 83).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask


# The following are not from DeepLearningExamples.
class VariancePredictor(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
            nn.Conv1d(d_inner, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
            nn.Linear(d_inner, 1),
        )

    def forward(self, vp_input):
        return self.layers(vp_input).squeeze(-1)


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hiddens, durations):
        """
        Expands the hidden states according to the duration target/prediction (depends on train vs inference).

        Args:
            hiddens: Hidden states of dimension (batch, time, emb_dim)
            durations: Timings for each frame of the hiddens, dimension (batch, time)
        """
        # Find max expanded length over batch elements for padding
        max_len = torch.max(torch.sum(durations, 1))

        out_list = []
        for x, d in zip(hiddens, durations):
            # For frame i of a single batch element x, repeats each the frame d[i] times.
            repeated = torch.cat([x[i].repeat(d[i], 1) for i in range(d.numel()) if d[i] != 0])
            repeated = F.pad(repeated, (0, 0, 0, max_len - repeated.shape[0]), "constant", value=0.0)
            out_list.append(repeated)

        out = torch.stack(out_list)


class DilatedResidualConvBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation, kernel_size):
        """
        Dilated residual convolutional block for the waveform decoder.
        residual_channels = input dimension. Input: (batch, residual_channels, time)
        """
        super().__init__()

        self.n_channels = residual_channels

        # Dilated conv
        padding = int((kernel_size * dilation - dilation) / 2)
        self.dilated_conv = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=(2 * self.n_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Pointwise conv for residual
        self.pointwise_conv_residual = nn.Conv1d(
            in_channels=self.n_channels, out_channels=residual_channels, kernel_size=1
        )

        # Pointwise conv for skip connection (this is separate from resids but not mentioned in the WaveNet paper)
        self.pointwise_conv_skip = nn.Conv1d(in_channels=self.n_channels, out_channels=skip_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.dilated_conv(x)
        out = nn.Tanh(out[:, : self.n_channels, :]) * torch.sigmoid(out[:, self.n_channels :, :])

        # Skip connection
        skip_out = self.pointwise_conv_skip(out)

        # Residual connection
        out = (out + residual) * torch.sqrt(0.5)

        return skip_out, out


def _conv_weight_norm(module):
    """
    Function to apply weight norm to only convolutional layers in the waveform decoder.
    """
    # TODO: Should convtranspose1d also be included?
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
        nn.utils.weight_norm(module)


class WaveformGenerator(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=1,
        trans_kernel_size=64,
        hop_size=256,
        n_layers=30,
        dilation_cycle=3,
        dilated_kernel_size=3,
        residual_channels=64,
        skip_channels=64,
    ):
        """
        Waveform generator for FastSpeech 2s, based on WaveNet and Parallel WaveGAN.
        """
        if n_layers // dilation_cycle != 0:
            logging.error(
                f"Number of layers in dilated residual convolution blocks should be divisible by dilation cycle."
                f" Have {n_layers} layers and cycle size {dilation_cycle}, which are not divisible."
            )

        self.n_layers = n_layers

        # Transposed 1D convolution to upsample slices of hidden reps to a longer audio length
        # TODO: double-check transposed conv args. -- kernel size in particular.
        #       The FastSpeech 2 paper says "filter size 64," Huihan's repo uses kernel_size=3.
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=residual_channels, kernel_size=trans_kernel_size, stride=hop_size,
        )

        # Repeated dilated residual convolution blocks
        self.dilated_res_conv_blocks = nn.ModuleList()
        dilation = 1

        for i in range(n_layers):
            self.dilated_res_conv_blocks.append(
                DilatedResidualConvBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    dilation=dilation,
                    kernel_size=dilated_kernel_size,
                )
            )
            # Increase dilation by a factor of 2 every {dilation_cycle}-layers.
            if (i + 1) % dilation_cycle == 0:
                dilation *= 2

        # Output activations and pointwise convolutions
        self.out_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),  # TODO: output dim here is a guess.
            nn.ReLU(),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1),
        )

        # Apply weight norm to conv layers
        self.apply(_conv_weight_norm)

    def forward(self, x, use_softmax=False):
        # Expand via upsampling
        x = self.transposed_conv(x)

        # Dilated conv blocks
        skip_outs = 0
        for i in range(self.n_layers):
            skip_out, x = self.dilated_res_conv_blocks[i](x)
            skip_outs += skip_out
        skip_outs *= torch.sqrt(1.0 / self.n_layers)

        # Output layers
        out = self.out_layers(skip_outs)

        if use_softmax:
            out = nn.Softmax(out, dim=1)

        return out


class WaveformDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_layers=10,
        kernel_size=3,
        conv_channels=64,
        conv_stride=1,
        relu_alpha=0.2,
    ):
        """
       Waveform discriminator for FastSpeech 2s, based on Parallel WaveGAN.
       """
        # Layers of non-causal dilated 1D convolutions and leaky ReLU
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        channels = conv_channels

        for i in range(n_layers - 1):
            # Dilated 1D conv
            dilation = i if i > 0 else 1
            padding = int((kernel_size * dilation - dilation) / 2)
            self.layers.append(
                nn.Conv1d(
                    in_channels=prev_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=conv_stride,
                )
            )
            prev_channels = channels

            # Leaky ReLU
            self.layers.append(nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

        # Last layer
        self.layer.append(
            nn.Conv1d(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,  # TODO: dilation=1 or (n_layers-2)?
                padding=int((kernel_size - 1) / 2),
            )
        )

        # Apply weight norm to conv layers
        self.apply(_conv_weight_norm)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
