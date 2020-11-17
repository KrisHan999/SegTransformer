import torch.nn as nn
import torch
from models.trasnformer import BasicTransformer
from models.padMultiAttn import PadTransformer


class Transformer(nn.Module):
    def __init__(self,
                 num_layers_encoder, num_layers_tokener, num_layers_decoder,
                 nhead_encoder, nhead_tokener, nhead_decoder,
                 d_model, dim_feedforward=2048, dropout=0.1, activation="relu",
                 with_background=True):
        super(Transformer, self).__init__()
        self.with_background = with_background
        self.transformer_encoder = BasicTransformer(
            num_layers=num_layers_encoder, d_model=d_model, nhead=nhead_encoder,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            use_key_pos=False, use_query_pos=False
        )
        self.transformer_tokener = BasicTransformer(
            num_layers=num_layers_tokener, d_model=d_model, nhead=nhead_tokener,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            use_key_pos=False, use_query_pos=False,
            update_key_value_using_query=False
        )
        self.transformer_decoder = BasicTransformer(
            num_layers=num_layers_decoder, d_model=d_model, nhead=nhead_decoder,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            use_key_pos=False, use_query_pos=False
        )

    def forward(self, query, key, value):
        tgt_from_encoder, attention_map_from_encoder = self.transformer_encoder(key=key, value=value, query=query)
        # if self.with_background:
        #     tgt_from_encoder_bg = tgt_from_encoder[-1:]
        #     tgt_from_encoder = tgt_from_encoder[:-1]
        tgt_from_tokener, attention_map_from_tokener = self.transformer_tokener(key=tgt_from_encoder,
                                                                                value=tgt_from_encoder,
                                                                                query=tgt_from_encoder)
        # if self.with_background:
        #     tgt_from_tokener = torch.cat([tgt_from_tokener, tgt_from_encoder_bg], dim=0)
        tgt_from_decoder, attention_map_from_decoder = self.transformer_decoder(key=tgt_from_tokener,
                                                                                value=tgt_from_tokener,
                                                                                query=key)
        return tgt_from_decoder, tgt_from_tokener, attention_map_from_decoder
