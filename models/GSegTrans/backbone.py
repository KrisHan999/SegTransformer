import torch
import torch.nn as nn
from models.backbone import DoubleConv3x3, Encoder
import numpy as np
from models.trasnformer import BasicTransformer


class Backbone(nn.Module):
    def __init__(self, n_class, n_channel, start_channel=32, n_decode_channel=256, normalization='bn', activation='relu', num_groups=None):
        super(Backbone, self).__init__()
        self.channels = np.asarray([start_channel * 2 ** i for i in range(5)])
        self.encoder = Encoder(n_channel=n_channel, start_channel=start_channel, normalization=normalization,
                               activation=activation, num_groups=num_groups)
        self.upsample5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        # self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        # self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #
        # self.fusion_conv = DoubleConv3x3(ch_in=self.channels.sum(), ch_out=n_decode_channel,
        #                                  normalization=normalization, activation=activation, num_groups=num_groups)
        n_decode_channel = self.channels[-1]
        self.decoder_transformer = BasicTransformer(2, d_model=n_decode_channel, nhead=4, dim_feedforward=512,
                                                    dropout=0.1, activation=activation, use_key_pos=False,
                                                    use_query_pos=False)

        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=n_decode_channel)

    def forward(self, x):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = self.encoder(x)
        # enc_out_up_1 = enc_out_1
        # enc_out_up_2 = self.upsample2(enc_out_2)
        # enc_out_up_3 = self.upsample3(enc_out_3)
        # enc_out_up_4 = self.upsample4(enc_out_4)
        #
        # fusion_input = torch.cat([enc_out_up_1,
        #                           enc_out_up_2,
        #                           enc_out_up_3,
        #                           enc_out_up_4,
        #                           enc_out_up_5],
        #                          dim=1)
        #
        # fusion_out = self.fusion_conv(fusion_input)

        n, c, h, w = enc_out_5.shape

        input_token_embedding = self.token_embedding.weight.unsqueeze(1).repeat(1, n, 1)
        src = enc_out_5.flatten(2).permute(2, 0, 1)
        _, attention_map = self.decoder_transformer(key=src, value=src, query=input_token_embedding)
        attention_map = attention_map.view(n, -1, h, w)
        attention_map = torch.clamp(attention_map, min=0, max=1)
        attention_map = self.upsample5(attention_map)
        return [attention_map]



