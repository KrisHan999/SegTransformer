import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from models.trasnformer import _get_activation_fn


class PadTransformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 add_noise_term=True):
        super(PadTransformer, self).__init__()
        self.layers = nn.ModuleList([PadTransformerLayer(d_model, nhead,
                                                         dim_feedforward=dim_feedforward,
                                                         dropout=dropout,
                                                         activation=activation,
                                                         add_noise_term=add_noise_term)
                                     for i in range(num_layers)])

    def forward(self, key, value, query):
        tgt = query
        attention_map = None
        for layer in self.layers:
            res = layer(key=key, value=value, query=tgt)
            tgt = res['tgt']
            attention_map = res['attn_map']

        return tgt, attention_map


class PadTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", add_noise_term=True):
        super(PadTransformerLayer, self).__init__()
        self.add_noise_term = add_noise_term
        self.d_model = d_model
        self.self_attn = PadMultiAttn(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, key: Tensor, value: Tensor, query: Tensor):
        """
        In this layer, token is the src, pixel feature is the query.
        :param src: [S, N, E]
        :param query: [L, N, E]
        :return:
            tgt: [L, N, E]
            attention_map: None or [N, L, S]
        """
        if key.shape[2] != self.d_model or value.shape[2] != self.d_model or query.shape[2] != self.d_model:
            raise ValueError(
                f"Error in dimension checking - {self.d_model}, {key.shape[2]},  {value.shape[2]}, {query.shape[2]}")
        tgt, attention_map, seg_attn = self.self_attn(query=query, key=key, value=value, add_noise_term=self.add_noise_term)
        if self.add_noise_term:
            assert attention_map.shape[-1] == key.shape[0] + 1, "fail to add noise term"
            assert seg_attn.shape[-1] == key.shape[0] + 1, "fail to add noise term"
            attention_map = attention_map[:, :, :-1]
            seg_attn = seg_attn[:, :, :-1]
        tgt = query + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt_2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt_2)
        tgt = self.norm2(tgt)

        res = {
            'tgt': tgt,
            'attn_map': seg_attn
        }
        return res


class PadMultiAttn(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(PadMultiAttn, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        self.bias_k = self.bias_v = None
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, need_weights=True, add_noise_term=False):
        return multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, need_weights=need_weights, add_noise_term=add_noise_term)


def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 need_weights: bool = True,
                                 add_noise_term: bool = False,
                                 ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if torch.equal(query, key) and torch.equal(key, value):
        # self-attention
        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif torch.equal(key, value):
        # encoder-decoder attention
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        if key is None:
            assert value is None
            k = None
            v = None
        else:

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

    else:
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(value, _w, _b)

    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if add_noise_term:
        src_len += 1
        attn_output_weights = torch.cat([attn_output_weights, torch.zeros(bsz * num_heads, tgt_len, 1).to(attn_output_weights.device)], dim=-1)
        v = torch.cat([v, torch.randn(bsz * num_heads, 1, head_dim).to(v.device)], dim=1)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    seg_attn = F.sigmoid(attn_output_weights)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        seg_attn = seg_attn.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, seg_attn.sum(dim=1) / num_heads
    else:
        return attn_output, None, None
