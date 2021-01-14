import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom
from numba import jit, prange


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            # log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear',
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


# class GetPrior(torch.nn.Module):
#     def forward(self, mel_length, text_length):
#         prior = []
#         for i, _ in enumerate(mel_length):
#             prior.append(beta_binomial_prior_distribution(text_length[i].cpu().numpy(), mel_length[i].cpu().numpy()))
#         import ipdb

#         ipdb.set_trace()


class ConvAttention(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=256,
        n_att_channels=80,
        temperature=1.0,
        align_query_enc_type='3xconv',
        use_query_proj=True,
    ):
        super().__init__()
        # self.temperature = temperature
        # self.att_scaling_factor = np.sqrt(n_att_channels)
        # self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        # self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        self.attn_proj = torch.nn.Conv2d(n_att_channels, 1, kernel_size=1)
        self.align_query_enc_type = align_query_enc_type
        self.use_query_proj = bool(use_query_proj)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.align_query_enc_type = align_query_enc_type

        if align_query_enc_type == "inv_conv":
            self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        elif align_query_enc_type == "3xconv":
            self.query_proj = nn.Sequential(
                ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
            )
        else:
            raise ValueError("Unknown query encoder type specified")

    # def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data, recurrent_model):
    #     """Sorts input data by previded ordering (and un-ordering) and runs the
    #     packed data through the recurrent model

    #     Args:
    #         sorted_idx (torch.tensor): 1D sorting index
    #         unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
    #         lens: lengths of input data (sorted in descending order)
    #         padded_data (torch.tensor): input sequences (padded)
    #         recurrent_model (nn.Module): recurrent model to run data through
    #     Returns:
    #         hidden_vectors (torch.tensor): outputs of the RNN, in the original,
    #         unsorted, ordering
    #     """

    #     # sort the data by decreasing length using provided index
    #     # we assume batch index is in dim=1
    #     padded_data = padded_data[:, sorted_idx]
    #     padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
    #     hidden_vectors = recurrent_model(padded_data)[0]
    #     hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
    #     # unsort the results at dim=1 and return
    #     hidden_vectors = hidden_vectors[:, unsort_idx]
    #     return hidden_vectors

    # def encode_query(self, query, query_lens):
    #     query = query.permute(2, 0, 1)  # seq_len, batch, feature dim
    #     lens, ids = torch.sort(query_lens, descending=True)
    #     original_ids = [0] * lens.size(0)
    #     for i in range(len(ids)):
    #         original_ids[ids[i]] = i

    #     query_encoded = self.run_padded_sequence(ids, original_ids, lens, query, self.query_lstm)
    #     query_encoded = query_encoded.permute(1, 2, 0)
    #     return query_encoded

    def forward(self, queries, keys, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc, since we
        only need this during training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1
        """

        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        if self.use_query_proj:
            if self.align_query_enc_type == "inv_conv":
                queries_enc, log_det_W = self.query_proj(queries)
            elif self.align_query_enc_type == "3xconv":
                queries_enc = self.query_proj(queries)
                log_det_W = 0.0
            else:
                queries_enc, log_det_W = self.query_proj(queries)
        else:
            queries_enc, log_det_W = queries, 0.0

        # different ways of computing attn, one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -0.0005 * attn.sum(1, keepdim=True)  # compute log likelihood from a gaussian
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        return attn


class SingleHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(
        self, n_mel_channels=80, n_text_channels=256, n_att_channels=80, attn_score_dropout=0.0, attn_layer_dropout=0.0
    ):
        super().__init__()

        self.hidden_size = n_att_channels
        self.attn_scale = math.sqrt(math.sqrt(self.hidden_size))

        self.query_net = nn.Linear(n_mel_channels, self.hidden_size)
        self.key_net = nn.Linear(n_text_channels, self.hidden_size)
        self.value_net = nn.Linear(n_text_channels, self.hidden_size)
        self.out_projection = nn.Linear(self.hidden_size, 256)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask, prior=None, binarize=False, in_len=None, out_len=None):
        # Mel T1, Text T2, Text
        query = self.query_net(queries) / self.attn_scale
        key = self.key_net(keys) / self.attn_scale
        value = self.value_net(values)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # B x T1 x T2
        if attention_mask is not None:
            attention_scores.masked_fill_(~attention_mask.unsqueeze(1), -float("inf"))
        if prior is not None:
            attention_scores = torch.log_softmax(attention_scores, dim=-1)
            attention_scores = attention_scores + torch.log(prior + 1e-8)
        soft_attn = torch.softmax(attention_scores, dim=-1)
        attention_probs = soft_attn
        if binarize:
            b_size = soft_attn.shape[0]
            with torch.no_grad():
                attn_cpu = soft_attn.data.cpu().numpy()
                hard_attn = torch.zeros_like(soft_attn)
                for ind in range(b_size):
                    hard_attn_calc = mas(attn_cpu[ind, : out_len[ind], : in_len[ind]], width=1)
                    hard_attn[ind, : out_len[ind], : in_len[ind]] = torch.tensor(
                        hard_attn_calc, device=soft_attn.get_device()
                    )
            attention_probs = hard_attn
        attention_probs = self.attn_dropout(attention_probs)
        context = torch.matmul(attention_probs, value)  # B T1 T2 x B T2 H
        # context = context.permute(0, 2, 1, 3).contiguous()
        # new_context_shape = context.size()[:-2] + (self.hidden_size,)
        # context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        return output_states, attention_probs, soft_attn


class ABLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()


class Loss2(torch.nn.Module):  # CTCLoss
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            cost_total += ctc_cost
        cost = cost_total / attn_logprob.shape[0]
        return cost


# def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=0.05):
#     P = phoneme_count
#     M = mel_count
#     x = np.arange(0, P)
#     mel_text_probs = []
#     for i in range(1, M + 1):
#         # TODO(ROhan): check if i or i+1
#         a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
#         # print(a, b)
#         rv = betabinom(P, a, b)
#         mel_i_prob = rv.pmf(x)
#         mel_text_probs.append(mel_i_prob)
#     return torch.tensor(np.array(mel_text_probs))


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


class Loss3(torch.nn.Module):  # AttentionBinarizationLoss
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()
