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


class LocalAttnTransformer(nn.Module):
    def __init__(self, kernel_size, padding, num_layers, d_model, nhead,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(LocalAttnTransformer, self).__init__()
        self.layers = nn.ModuleList([LocalAttnTransformerLayer(kernel_size, padding, d_model, nhead,
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout,
                                                               activation=activation)
                                     for i in range(num_layers)])

    def forward(self, feature):
        attention_map = None
        for layer in self.layers:
            res = layer(feature)
            feature = res['feature']
            attention_map = res['attn_map']

        return feature, attention_map


class LocalAttnTransformerLayer(nn.Module):
    def __init__(self, kernel_size, padding, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(LocalAttnTransformerLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = LocalAttn(kernel_size=kernel_size, padding=padding,
                                   embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(num_features=d_model)
        self.norm2 = nn.BatchNorm2d(num_features=d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, feature: Tensor):
        """
        In this layer, token is the src, pixel feature is the query.
        :param feature: [N, C, H, W]
        :return:
            tgt: [N, C, H, W]
            attention_map: None or [N, K, H, W]
        """
        out, weights = self.self_attn(feature=feature)
        out = feature + self.dropout1(out)
        out = self.norm1(out)
        out = out.permute(0, 2, 3, 1)
        out_2 = self.linear2(self.dropout2(self.activation(self.linear1(out))))
        out = out.permute(0, 3, 1, 2)
        out_2 = out_2.permute(0, 3, 1, 2)
        out = out + self.dropout3(out_2)
        out = self.norm2(out)

        res = {
            'feature': out,
            'attn_map': weights
        }
        return res


class LocalAttn(Module):
    def __init__(self, kernel_size, padding, embed_dim, num_heads, dropout=0.1):
        super(LocalAttn, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
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

    def forward(self, feature, need_weights=True):
        return local_multi_head_attention_forward(
            self.kernel_size, self.padding,
            feature, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, need_weights=need_weights)


def local_multi_head_attention_forward(kernel_size: int,
                                       padding: int,
                                       feature: Tensor,
                                       embed_dim_to_check: int,
                                       num_heads: int,
                                       in_proj_weight: Tensor,
                                       in_proj_bias: Tensor,
                                       dropout_p: float,
                                       out_proj_weight: Tensor,
                                       out_proj_bias: Tensor,
                                       training: bool = True,
                                       need_weights: bool = True
                                       ) -> Tuple[Tensor, Optional[Tensor]]:
    n, c, h, w = feature.shape
    feature = feature.permute(0, 2, 3, 1)  # n, h, w, c
    assert c == embed_dim_to_check
    head_dim = c // num_heads
    assert head_dim * num_heads == c, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    q, k, v = F.linear(feature, in_proj_weight, in_proj_bias).permute(0, 3, 1, 2).chunk(3, dim=1)  # n, c, h, w

    k = F.unfold(k, padding=padding, kernel_size=kernel_size)  # n, c*factor(L), h', w'
    k = k.view(n, c, kernel_size * kernel_size, h, w)  # view as this format!!!
    v = F.unfold(v, padding=padding, kernel_size=kernel_size)
    v = v.view(n, c, kernel_size * kernel_size, h, w)

    q = q * scaling
    q = q.contiguous().view(n * num_heads, head_dim, h, w)
    k = k.contiguous().view(n * num_heads, head_dim, kernel_size * kernel_size, h, w)
    v = v.contiguous().view(n * num_heads, head_dim, kernel_size * kernel_size, h, w)

    weights = torch.einsum('bckhw, bchw -> bkhw', k, q)
    weights = weights.softmax(dim=1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    out = torch.einsum('bckhw, bkhw -> bchw', v, weights)
    assert list(out.size()) == [n * num_heads, head_dim, h, w]
    out = out.view(n, num_heads, head_dim, h, w).flatten(start_dim=1, end_dim=2).permute(0, 2, 3, 1)  # n, h, w, c
    out = F.linear(out, out_proj_weight, out_proj_bias).permute(0, 3, 1, 2)

    if need_weights:
        weights = weights.view(n, num_heads, kernel_size * kernel_size, h, w)
        return out, weights.sum(dim=1) / num_heads
    else:
        return out, None
