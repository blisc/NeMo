import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from nemo.utils import logging
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
# from nemo.collections.nlp.modules.common.transformer.transformer_modules import (
#     MultiHeadAttention as MultiHeadAttention2,
# )


class Bmm(nn.Module):
    """ Required for manual fp16 casting. If not using amp_opt_level='O2', just use torch.bmm.
    """

    def forward(self, a, b):
        return torch.bmm(a, b)


class FFTBlocksWithEncDecAttn(nn.Module):
    def __init__(
        self,
        name,
        max_seq_len,
        n_layers=4,
        n_head=2,
        d_k=64,
        d_v=64,
        d_model=256,
        d_inner=1024,
        d_word_vec=256,
        fft_conv1d_kernel_1=9,
        fft_conv1d_kernel_2=1,
        fft_conv1d_padding_1=4,
        fft_conv1d_padding_2=0,
        dropout=0.2,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_word_vec = d_word_vec
        self.d_inner = d_inner
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.fused_layernorm = fused_layernorm
        self.name = name

        n_position = max_seq_len + 1
        self.position = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList()
        self.layer_stack.append(
            FFTBlockAttn(
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                fft_conv1d_kernel_1=fft_conv1d_kernel_1,
                fft_conv1d_kernel_2=fft_conv1d_kernel_2,
                fft_conv1d_padding_1=fft_conv1d_padding_1,
                fft_conv1d_padding_2=fft_conv1d_padding_2,
                dropout=dropout,
                fused_layernorm=fused_layernorm,
                use_amp=use_amp,
                name="{}.layer_stack.0".format(self.name),
            )
        )
        for i in range(1, n_layers):
            self.layer_stack.append(
                FFTBlock(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    fft_conv1d_kernel_1=fft_conv1d_kernel_1,
                    fft_conv1d_kernel_2=fft_conv1d_kernel_2,
                    fft_conv1d_padding_1=fft_conv1d_padding_1,
                    fft_conv1d_padding_2=fft_conv1d_padding_2,
                    dropout=dropout,
                    fused_layernorm=fused_layernorm,
                    use_amp=use_amp,
                    name="{}.layer_stack.{}".format(self.name, i),
                )
            )

    def forward(self, seq, lengths, encoder_keyvalue, enc_len):

        non_pad_mask = get_mask_from_lengths(lengths, max_len=seq.size(1))

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)  # (b, t, t)
        non_pad_mask = non_pad_mask.unsqueeze(-1)

        # -- Forward
        seq_length = seq.size(1)
        if seq_length > self.max_seq_len:
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_seq_len}"
            )
        position_ids = torch.arange(start=0, end=0 + seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(seq.size(0), -1)
        pos_enc = self.position(position_ids) * non_pad_mask
        output = seq + pos_enc

        enc_dec_layer = self.layer_stack[0]
        enc_mask = get_attn_key_pad_mask(seq_k=get_mask_from_lengths(enc_len), seq_q=non_pad_mask)
        enc_mask *= -float("Inf")
        output, attn = enc_dec_layer(output, encoder_keyvalue, non_pad_mask, slf_attn_mask, enc_mask)

        for layer in self.layer_stack[1:]:
            output, _ = layer(output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        return output, attn


class FFTBlocks(nn.Module):
    def __init__(
        self,
        name,
        max_seq_len,
        n_layers=4,
        n_head=2,
        d_k=64,
        d_v=64,
        d_model=256,
        d_inner=1024,
        d_word_vec=256,
        fft_conv1d_kernel_1=9,
        fft_conv1d_kernel_2=1,
        fft_conv1d_padding_1=4,
        fft_conv1d_padding_2=0,
        dropout=0.2,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_word_vec = d_word_vec
        self.d_inner = d_inner
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.fused_layernorm = fused_layernorm
        self.name = name

        n_position = max_seq_len + 1
        self.position = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    fft_conv1d_kernel_1=fft_conv1d_kernel_1,
                    fft_conv1d_kernel_2=fft_conv1d_kernel_2,
                    fft_conv1d_padding_1=fft_conv1d_padding_1,
                    fft_conv1d_padding_2=fft_conv1d_padding_2,
                    dropout=dropout,
                    fused_layernorm=fused_layernorm,
                    use_amp=use_amp,
                    name="{}.layer_stack.{}".format(self.name, i),
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, seq, lengths, return_attns=False, acts=None):

        slf_attn_list = []
        non_pad_mask = get_mask_from_lengths(lengths, max_len=seq.size(1))

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)  # (b, t, t)
        non_pad_mask = non_pad_mask.unsqueeze(-1)

        # -- Forward
        seq_length = seq.size(1)
        if seq_length > self.max_seq_len:
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_seq_len}"
            )
        position_ids = torch.arange(start=0, end=0 + seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(seq.size(0), -1)
        pos_enc = self.position(position_ids) * non_pad_mask
        output = seq + pos_enc

        if acts is not None:
            acts["act.{}.add_pos_enc".format(self.name)] = output

        for i, layer in enumerate(self.layer_stack):
            output, slf_attn = layer(output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask, acts=acts)
            if return_attns:
                slf_attn_list += [slf_attn]

            if acts is not None:
                acts['act.{}.layer_stack.{}'.format(self.name, i)] = output

        return output, non_pad_mask


class FFTBlockAttn(torch.nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        fft_conv1d_kernel_1,
        fft_conv1d_kernel_2,
        fft_conv1d_padding_1,
        fft_conv1d_padding_2,
        dropout,
        name,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.name = name
        self.fused_layernorm = fused_layernorm

        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            name="{}.slf_attn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

        self.enc_dec_attn = MultiHeadAttention2(hidden_size=d_model, num_attention_heads=1, attn_layer_dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            fft_conv1d_kernel_1=fft_conv1d_kernel_1,
            fft_conv1d_kernel_2=fft_conv1d_kernel_2,
            fft_conv1d_padding_1=fft_conv1d_padding_1,
            fft_conv1d_padding_2=fft_conv1d_padding_2,
            dropout=dropout,
            name="{}.pos_ffn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

    def forward(self, decoder_input, encoder_keyvalue, non_pad_mask=None, slf_attn_mask=None, enc_mask=None):
        output, slf_attn = self.slf_attn(decoder_input, mask=slf_attn_mask)
        output *= non_pad_mask.to(output.dtype)

        output, enc_dec_attn = self.enc_dec_attn(output, encoder_keyvalue, encoder_keyvalue, attention_mask=enc_mask)
        output *= non_pad_mask.to(output.dtype)

        output = self.pos_ffn(output)
        output *= non_pad_mask.to(output.dtype)

        return output, enc_dec_attn


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        fft_conv1d_kernel_1,
        fft_conv1d_kernel_2,
        fft_conv1d_padding_1,
        fft_conv1d_padding_2,
        dropout,
        name,
        fused_layernorm=False,
        use_amp=False,
    ):
        super(FFTBlock, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.name = name
        self.fused_layernorm = fused_layernorm

        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            name="{}.slf_attn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

        self.pos_ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            fft_conv1d_kernel_1=fft_conv1d_kernel_1,
            fft_conv1d_kernel_2=fft_conv1d_kernel_2,
            fft_conv1d_padding_1=fft_conv1d_padding_1,
            fft_conv1d_padding_2=fft_conv1d_padding_2,
            dropout=dropout,
            name="{}.pos_ffn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

    def forward(self, _input, non_pad_mask=None, slf_attn_mask=None, acts=None):
        output, slf_attn = self.slf_attn(_input, mask=slf_attn_mask, acts=acts)

        output *= non_pad_mask.to(output.dtype)

        output = self.pos_ffn(output, acts=acts)
        output *= non_pad_mask.to(output.dtype)

        return output, slf_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout, name, fused_layernorm=False, use_amp=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.name = name
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        d_out = d_k + d_k + d_v
        self.linear = nn.Linear(d_model, n_head * d_out)
        nn.init.xavier_normal_(self.linear.weight)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), name="{}.scaled_dot".format(self.name)
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, acts=None):
        bs, seq_len, _ = x.size()

        residual = x

        d_out = self.d_k + self.d_k + self.d_v
        x = self.linear(x)  # (b, t, n_heads * h)

        if acts is not None:
            acts['act.{}.linear'.format(self.name)] = x

        x = x.view(bs, seq_len, self.n_head, d_out)  # (b, t, n_heads, h)
        x = x.permute(2, 0, 1, 3).contiguous().view(self.n_head * bs, seq_len, d_out)  # (n * b, t, h)

        q = x[..., : self.d_k]  # (n * b, t, d_k)
        k = x[..., self.d_k : 2 * self.d_k]  # (n * b, t, d_k)
        v = x[..., 2 * self.d_k :]  # (n * b, t, d_k)

        mask = mask.repeat(self.n_head, 1, 1)  # (b, t, h) -> (n * b, t, h)

        output, attn = self.attention(q, k, v, mask=mask, acts=acts)

        output = output.view(self.n_head, bs, seq_len, self.d_v)  # (n, b, t, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, self.n_head * self.d_v)  # (b, t, n * d_k)

        if acts is not None:
            acts['act.{}.scaled_dot'.format(self.name)] = output

        output = self.fc(output)

        output = self.dropout(output)

        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, name=None):
        super().__init__()

        self.temperature = temperature
        self.name = name

        self.bmm1 = Bmm()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.bmm2 = Bmm()

    def forward(self, q, k, v, mask=None, acts=None):

        attn = self.bmm1(q, k.transpose(1, 2))

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -65504)

        attn = self.softmax(attn)

        attn = self.dropout(attn)

        output = self.bmm2(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(
        self,
        d_in,
        d_hid,
        fft_conv1d_kernel_1,
        fft_conv1d_kernel_2,
        fft_conv1d_padding_1,
        fft_conv1d_padding_2,
        dropout,
        name,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.name = name
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=fft_conv1d_kernel_1, padding=fft_conv1d_padding_1)

        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=fft_conv1d_kernel_2, padding=fft_conv1d_padding_2)

        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, acts=None):
        residual = x

        output = x.transpose(1, 2)
        output = self.w_1(output)

        if acts is not None:
            acts['act.{}.conv1'.format(self.name)] = output

        output = F.relu(output)
        output = self.w_2(output)
        if acts is not None:
            acts['act.{}.conv2'.format(self.name)] = output

        output = output.transpose(1, 2)
        output = self.dropout(output)

        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        if self.fused_layernorm and self.use_amp:
            from torch.cuda import amp

            with amp.autocast(enabled=False):
                output = output.float()
                output = self.layer_norm(output)
                output = output.half()
        else:
            output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)  # (b, t)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # (b, t, t)

    return padding_mask


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).unsqueeze(-1)


def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.long)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).long()
    return result


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(
        self,
        f0_min,
        f0_max,
        energy_min,
        energy_max,
        n_bins,
        encoder_hidden,
        variance_predictor_filter_size,
        variance_predictor_kernel_size,
        variance_predictor_dropout,
        fused_layernorm=False,
        use_amp=False,
    ):
        super(VarianceAdaptor, self).__init__()

        self.f0_min = f0_min
        self.f0_max = f0_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.n_bins = n_bins
        self.encoder_hidden = encoder_hidden
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        self.duration_predictor = VariancePredictor(
            encoder_hidden,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            fused_layernorm,
            use_amp,
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(
            encoder_hidden,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            fused_layernorm,
            use_amp,
        )
        self.energy_predictor = VariancePredictor(
            encoder_hidden,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            fused_layernorm,
            use_amp,
        )

        self.pitch_bins = nn.Parameter(
            torch.exp(torch.linspace(np.log(self.f0_min), np.log(self.f0_max), self.n_bins - 1))
        )
        self.energy_bins = nn.Parameter(torch.linspace(self.energy_min, self.energy_max, self.n_bins - 1))
        self.pitch_embedding = nn.Embedding(self.n_bins, self.encoder_hidden)
        self.energy_embedding = nn.Embedding(self.n_bins, self.encoder_hidden)

    def forward(
        self, x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x = self.length_regulator(_input=x, duration=duration_target)
            mel_length = torch.sum(duration_target, dim=1)
        else:
            duration_rounded = torch.clamp_min(torch.exp(log_duration_prediction) - 1, 0).long()
            if not torch.sum(duration_rounded, dim=1).bool().all():
                logging.error("Duration prediction failed on this batch. Settings to 1s")
                duration_rounded += 1
                logging.debug(duration_rounded)
            if torch.ge(torch.sum(duration_rounded, dim=1), 2048).any():
                logging.error("Duration prediction was too high this batch. Clamping further")
                length = duration_rounded.size(1)
                duration_rounded = torch.clamp(duration_rounded, max=2048 // length)
            x = self.length_regulator(_input=x, duration=duration_rounded)
            mel_length = torch.sum(duration_rounded, dim=1)

        mel_mask = get_mask_from_lengths(mel_length).unsqueeze(-1)

        # pos = pos.to(x.device)
        # mel_mask = mel_mask.to(x.device)
        pitch_prediction = self.pitch_predictor(x, mel_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_bucketize = bucketize(pitch_prediction, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_bucketize)
        energy_prediction = self.energy_predictor(x, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(bucketize(energy_target, self.energy_bins))
        else:
            energy_embedding = self.energy_embedding(bucketize(energy_prediction, self.energy_bins))

        x = x + pitch_embedding + energy_embedding

        return x, mel_length, log_duration_prediction, pitch_prediction, energy_prediction


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, _input, duration):
        output = []
        # TODO: parallelize the loop.
        for i in range(_input.size(0)):
            repeats = duration[i].float()
            repeats = torch.round(repeats).long()
            torch_repeat = torch.repeat_interleave(_input[i], repeats, dim=0)
            output.append(torch_repeat)

        output = pad_sequence(output, batch_first=True)

        return output


class VariancePredictor(nn.Module):
    """ Variance Predictor """

    def __init__(self, input_size, filter_size, kernel, dropout, fused_layernorm=False, use_amp=False):
        super(VariancePredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.dropout = dropout
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        self.conv1d_1 = nn.Conv1d(self.input_size, self.filter_size, kernel_size=self.kernel, padding=1)
        self.relu_1 = nn.ReLU()
        self.layer_norm_1 = nn.LayerNorm(self.filter_size)

        self.dropout_1 = nn.Dropout(self.dropout)

        self.conv1d_2 = nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)
        self.relu_2 = nn.ReLU()

        self.layer_norm_2 = nn.LayerNorm(self.filter_size)

        self.dropout_2 = nn.Dropout(self.dropout)

        self.linear_layer = nn.Linear(self.filter_size, 1, bias=True)

    def forward(self, _input, input_mask):
        _input *= input_mask.to(_input.dtype)

        out = self.conv1d_1(_input.transpose(1, 2)).transpose(1, 2)
        out = self.relu_1(out)

        if self.fused_layernorm and self.use_amp:
            from torch.cuda import amp

            with amp.autocast(enabled=False):
                out = out.float()
                out = self.layer_norm_1(out)
                out = out.half()
        else:
            out = self.layer_norm_1(out)
        out = self.dropout_1(out)

        out = self.conv1d_2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu_2(out)
        if self.fused_layernorm and self.use_amp:
            from torch.cuda import amp

            with amp.autocast(enabled=False):
                out = out.float()
                out = self.layer_norm_2(out)
                out = out.half()
        else:
            out = self.layer_norm_2(out)
        out = self.dropout_2(out)

        out = self.linear_layer(out)

        out *= input_mask.to(out.dtype)
        out = out.squeeze(-1)

        return out


class WaveDecoder(nn.Module):
    """ E2E WaveDecoder """

    def __init__(
        self,
        input_hidden,
        wavedecoder_generator_conv_kernel_size,
        wavedecoder_generator_conv_blocks,
        wavedecoder_generator_stacks,
        wavedecoder_generator_transposed_filter_size,
        wavedecoder_generator_gate_channels,
        wavedecoder_generator_skip_channel_size,
        wavedecoder_generator_residual_channel_size,
        dropout,
        bias,
        wavedecoder_discriminator_conv_kernel_size,
        wavedecoder_discriminator_conv_blocks,
        wavedecoder_discriminator_conv_channels,
        wavedecoder_discriminator_relu_alpha,
    ):
        super(WaveDecoder, self).__init__()
        self.generator = WaveNetBasedGenerator(
            input_hidden=input_hidden,
            kernel_size=wavedecoder_generator_conv_kernel_size,
            layers=wavedecoder_generator_conv_blocks,
            stacks=wavedecoder_generator_stacks,
            residual_channels=wavedecoder_generator_residual_channel_size,
            gate_channels=wavedecoder_generator_gate_channels,
            skip_channels=wavedecoder_generator_skip_channel_size,
            unsample_filter_size=wavedecoder_generator_transposed_filter_size,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, hid_seq, mask):
        hid_seq = hid_seq.transpose(1, 2)
        batch_size, _, batch_length = hid_seq.size()
        y_hat, y_start = self.generator(hid_seq, mask)
        y_hat = y_hat.squeeze(1)
        return y_hat, y_start


class WaveNetBasedGenerator(nn.Module):
    """ WaveNet based generator """

    def __init__(
        self,
        input_hidden=256,
        kernel_size=3,
        layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        unsample_filter_size=64,
        dropout=0.0,
        bias=True,
        freq_axis_kernel_size=3,
    ):
        super(WaveNetBasedGenerator, self).__init__()
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define the unsample network

        self.upsample_layers = torch.nn.ModuleList()
        for i in range(4):
            if i == 0:
                m = nn.ConvTranspose1d(
                    in_channels=input_hidden,
                    out_channels=unsample_filter_size,
                    kernel_size=4,
                    padding=0,
                    dilation=1,
                    stride=4,
                )  # np.prod(stride) = hop_size, and there is only 1 layer of ConvTranspose1d
            else:
                m = nn.ConvTranspose1d(
                    in_channels=unsample_filter_size,
                    out_channels=unsample_filter_size,
                    kernel_size=4,
                    padding=0,
                    dilation=1,
                    stride=4,
                )  # np.prod(stride) = hop_size, and there is only 1 layer of ConvTranspose1d
            m.weight.data.fill_(1.0 / unsample_filter_size)
            m.bias.data.zero_()
            self.upsample_layers += [m]

        # assuming we use [0, 1] scaled features
        # this should avoid non-negative upsampling output

        self.first_conv = Conv1d1x1(1, residual_channels, bias=True)
        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=unsample_filter_size,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = Conv1d1x1(skip_channels, 1, bias=True)

    def forward(self, _input, mask):
        # audio clipping
        if self.training:
            batch_size, _, _ = _input.size()
            mask = mask.transpose(1, 2)
            instances = list()
            starts = list()
            for i in range(batch_size):
                instance_length = int(torch.masked_select(_input[i], mask[i]).size()[0] / 256)
                y_start = random.randint(0, max(0, instance_length - 84 - 1))
                instances.append(_input[i, :, y_start : y_start + 84])
                starts.append(y_start + 2)
            _input = torch.stack(instances, dim=0)
            c = _input
            for f in self.upsample_layers:
                c = f(c)
            c = c[..., 512:-512]
        else:
            starts = [0]
            c = _input
            for f in self.upsample_layers:
                c = f(c)

        batch_size, _, batch_length = c.size()
        x = c.new_zeros(batch_size, 1, batch_length).normal_()
        # x = x.to(c.device)
        # starts = torch.tensor(starts).to(c.device)
        starts = torch.tensor(starts)

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        x = self.last_conv_layers(x)

        return x, starts


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=64,
        dropout=0.0,
        dilation=1,
        bias=True,
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        # no future time stamps available
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        padding = (kernel_size - 1) // 2 * dilation

        # dilation conv
        self.conv = nn.Conv1d(
            residual_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class Discriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=10,
        conv_channels=64,
        bias=True,
        relu_alpha=0.2,
        use_amp=False,
    ):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(Discriminator, self).__init__()
        self.use_amp = use_amp
        self.discriminator = ParallelWaveGANDiscriminator(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layers=layers,
            conv_channels=conv_channels,
            bias=bias,
            relu_alpha=relu_alpha,
        )

    def forward(self, y_hat, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        with amp.autocast(self.use_amp):
            y_hat = y_hat.unsqueeze(1)
            p_hat = self.discriminator(y_hat)
            y_hat, p_hat = y_hat.squeeze(1), p_hat.squeeze(1)
            y_disc, y_hat_disc = y.unsqueeze(1), y_hat.unsqueeze(1).detach()
            p_disc = self.discriminator(y_disc)
            p_hat_disc = self.discriminator(y_hat_disc)
            p_disc, p_hat_disc = p_disc.squeeze(1), p_hat_disc.squeeze(1)
        return p_hat, p_disc, p_hat_disc


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self, in_channels=1, out_channels=1, kernel_size=3, layers=10, conv_channels=64, bias=True, relu_alpha=0.2
    ):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        self.layers = layers
        self.kernel_size = kernel_size

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.LeakyReLU(inplace=True, negative_slope=relu_alpha),
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(conv_in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        use_amp=False,
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.use_amp = use_amp
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        with amp.autocast(self.use_amp):
            sc_loss = 0.0
            mag_loss = 0.0

            for f in self.stft_losses:
                sc_l, mag_l = f(x, y)
                sc_loss += sc_l
                mag_loss += mag_l

            sc_loss /= len(self.stft_losses)
            mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """

        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

        mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

        return sc_loss, mag_loss


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
