import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from typing import Optional


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Transformer(nn.Module):
    """
    Transformer is composed of three parts:
        TransformerEncoder: Update learned token given input pixel feature and generate position embedding for each
                            token.Each token is for one roi.
        TransformerToken:   Update token using mutual information, trying to discover the relation between different
                            token(ROI).
        TransformerDecoder: Update pixel feature using tokens which will give a more uniform feature map.
    """

    def __init__(self, need_generate_query_pos,
                 num_layers_encoder, num_layers_tokener, num_layers_decoder,
                 nhead_encoder, nhead_tokener, nhead_decoder,
                 d_model, d_pos,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_attention_map=True):
        super(Transformer, self).__init__()
        self.return_attention_map = return_attention_map
        self.encoder = TransformerEncoder(need_generate_query_pos, num_layers_encoder, d_model=d_model, d_pos=d_pos, nhead=nhead_encoder,
                                          dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                          return_attention_map=return_attention_map)
        self.tokener = TransformerToken(num_layers=num_layers_tokener, d_model=d_model, nhead=nhead_tokener,
                                        dropout=dropout, dim_feedforward=dim_feedforward, activation=activation)
        self.decoder = TransformerDecoder(num_layers=num_layers_decoder, d_model=d_model, nhead=nhead_decoder,
                                          dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                          return_attention_map=return_attention_map)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, pos_src: Tensor, query: Tensor, query_pos: Optional[Tensor]):
        """

        :param src: [S, N, E]
        :param pos_src: [S, N, E_p]
        :param query: [L, N, E]
        :return:
            tgt_from_decoder: [S, N, E]
            tgt_from_tokener: [L, N, E]
            pos_tgt_from_tokener: [L, N, E_p]
            attention_map_from_decoder: [N, S, L]
        """
        tgt_from_encoder, pos_tgt_from_encoder, attention_map_from_encoder = self.encoder(src, pos_src, query, query_pos)
        tgt_from_tokener, pos_tgt_from_tokener = self.tokener(tgt_from_encoder, pos_tgt_from_encoder)
        tgt_from_decoder, attention_map_from_decoder = self.decoder(tgt_from_tokener,
                                                                    pos_tgt_from_tokener,
                                                                    src,
                                                                    pos_src)
        return tgt_from_decoder, tgt_from_tokener, pos_tgt_from_tokener, attention_map_from_decoder


class TransformerEncoder(nn.Module):
    def __init__(self, need_generate_query_pos, num_layers, d_model, d_pos, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 return_attention_map=True):
        super(TransformerEncoder, self).__init__()
        self.need_generate_query_pos = need_generate_query_pos
        if need_generate_query_pos:
            self.start_layer = TransformerEncoderLayer(d_model, d_pos, nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation=activation,
                                                       start_layer=True,
                                                       return_attention_map=return_attention_map)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_pos, nhead,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout,
                                                             activation=activation,
                                                             start_layer=False,
                                                             return_attention_map=False)
                                     for i in range(num_layers)])

    def forward(self, src, pos_src, query, query_pos):
        tgt = query
        attention_map = None
        if self.need_generate_query_pos:
            tgt, pos_tgt, attention_map = self.start_layer(src, pos_src, query)
        else:
            pos_tgt = query_pos
        for layer in self.layers:
            tgt, pos_tgt, attention_map = layer(src, pos_src, tgt, pos_tgt)

        return tgt, pos_tgt, attention_map


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_pos, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 start_layer=True, return_attention_map=True):
        """

        :param d_model:
        :param nhead:
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param start_layer: If this layer is the start layer, it is used for computing the position embedding
        """
        super(TransformerEncoderLayer, self).__init__()
        self.start_layer = start_layer

        self.d_model = d_model
        if self.start_layer:
            d_model = d_model - d_pos
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = _get_activation_fn(activation)

        self.return_attention_map = return_attention_map

    def forward(self, src: Tensor, pos_src: Tensor, query: Tensor, query_pos: Optional[Tensor] = None):
        """
        In this layer, pixel feature is src and token are query
        :param src: [S, N, E]
        :param pos_src: [S, N, E_p]
        :param query: [L, N, E]
        :param query_pos: [L, N, E_p]
        :return:
            tgt: [L, N, E]
            pos_tgt: [L, N, E_p]
            attention_map: None or [N, L, S]
        """
        if src.shape[2] != query.shape[2] or src.shape[2] + pos_src.shape[2] != self.d_model:
            raise ValueError(
                f"Error in dimension checking - {self.d_model}, {src.shape[2]}, {pos_src.shape[2]}, {query.shape[2]}")
        if self.start_layer:
            q = query
            k = v = src
            tgt, attention_map = self.self_attn(query=q, key=k, value=v)  # [N, L, S]
            pos_tgt = attention_map.bmm(pos_src.transpose(0, 1))  # [N, L, E_p]
            pos_tgt = pos_tgt.transpose(0, 1)  # [L, N, E_p]
        else:
            E = query.shape[2]
            E_p = query_pos.shape[2]
            k = v = torch.cat([src, pos_src], dim=2)
            q = torch.cat([query, query_pos], dim=2)
            tgt, attention_map = self.self_attn(query=q, key=k, value=v)
            tgt = q + self.dropout1(tgt)
            tgt = self.norm1(tgt)
            tgt_2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt_2)
            tgt = self.norm2(tgt)
            tgt, pos_tgt = tgt.split([E, E_p], dim=2)

        if self.return_attention_map:
            return tgt, pos_tgt, attention_map
        else:
            return tgt, pos_tgt, None


class TransformerToken(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super(TransformerToken, self).__init__()
        self.layers = nn.ModuleList([TransformerTokenLayer(d_model=d_model,
                                                           nhead=nhead,
                                                           dropout=dropout,
                                                           dim_feedforward=dim_feedforward,
                                                           activation=activation) for i in range(num_layers)])

    def forward(self, raw_token, pos):
        output_token = raw_token
        output_pos = pos
        for layer in self.layers:
            output_token, output_pos = layer(output_token, output_pos)
        return output_token, output_pos


class TransformerTokenLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super(TransformerTokenLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = _get_activation_fn(activation)

    def forward(self, token, pos):
        """
        self-attention for tokens. Both k and q are tokens.
        :param token: [L, N, E]
        :param pos: [L, N, E_p]
        :return:
            token: [L, N, E]
            pos: [L, N, E_p]
        """
        L, N, E = token.shape
        L, N, E_p = pos.shape
        if self.d_model != (E + E_p):
            raise ValueError("TransformerToken has token and pos dimension different from d_model")
        q = k = v = torch.cat([token, pos], dim=2)
        tgt = self.self_attn(query=q, key=k, value=v)[0]
        tgt = q + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt_2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt_2)
        tgt = self.norm2(tgt)

        token, pos = tgt.split([E, E_p], dim=2)

        return token, pos


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 return_attention_map=True):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout,
                                                             activation=activation,
                                                             return_attention_map=return_attention_map)
                                     for i in range(num_layers)])

    def forward(self, src, pos_src, query, pos_query):
        tgt = query
        pos_tgt = pos_query
        attention_map = None
        for layer in self.layers:
            tgt, attention_map = layer(src, pos_src, tgt, pos_tgt)

        return tgt, attention_map


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 return_attention_map=True):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)
        self.return_attention_map = return_attention_map

    def forward(self, src: Tensor, pos_src: Tensor, query: Tensor, query_pos: Tensor):
        """
        In this layer, token is the src, pixel feature is the query.
        :param src: [S, N, E]
        :param pos_src: [S, N, E_p]
        :param query: [L, N, E]
        :param query_pos: [L, N, E_p]
        :return:
            tgt: [L, N, E]
            attention_map: None or [N, L, S]
        """
        if src.shape[2] != query.shape[2] or src.shape[2] + pos_src.shape[2] != self.d_model:
            raise ValueError(
                f"Error in dimension checking - {self.d_model}, {src.shape[2]}, {pos_src.shape[2]}, {query.shape[2]}")
        E = src.shape[2]
        E_p = pos_src.shape[2]
        k = v = torch.cat([src, pos_src], dim=2)
        q = torch.cat([query, query_pos], dim=2)
        tgt, attention_map = self.self_attn(query=q, key=k, value=v)
        tgt = q + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt_2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt_2)
        tgt = self.norm2(tgt)
        tgt, tgt_pos = tgt.split([E, E_p], dim=2)

        if self.return_attention_map:
            return tgt, attention_map
        else:
            return tgt, None
