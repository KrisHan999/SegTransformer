import torch
import torch.nn as nn
import numpy as np

from models.backbone import Encoder
from models.backbone import Conv, DoubleConv3x3, UpConv
from models.GSegTrans.transformer import Transformer
from models.segtransformer import TokenFN


class SegTransformerDecoder(nn.Module):
    def __init__(self, start_channel, n_class, input_dim=(512, 512), nhead=4,
                 normalization='bn', activation='relu', num_groups=None, with_background=True):
        super(SegTransformerDecoder, self).__init__()
        if input_dim is None:
            input_dim = [512, 512]
        self.start_channel = start_channel
        self.n_class = n_class
        channels = np.asarray([start_channel * 2 ** i for i in range(5)])  # 32, 62, 128, 256, 512

        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding_5 = nn.Embedding(num_embeddings=n_class, embedding_dim=channels[4])

        self.transformer5 = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                        d_model=channels[-1],
                                        dim_feedforward=32,
                                        with_background=with_background)

        self.transformer4 = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                        d_model=channels[-2],
                                        dim_feedforward=64,
                                        with_background=with_background)

        self.transformer3 = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                        d_model=channels[-3],
                                        dim_feedforward=128,
                                        with_background=with_background)

        self.transformer2 = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                        d_model=channels[-4],
                                        dim_feedforward=256,
                                        with_background=with_background)

        self.transformer1 = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                        d_model=channels[-5],
                                        dim_feedforward=512,
                                        with_background=with_background)

        self.token_fn_54 = TokenFN(channel_in=channels[-1], channel_out=channels[-2], dim_feedforward=channels[-1])
        self.token_fn_43 = TokenFN(channel_in=channels[-2], channel_out=channels[-3], dim_feedforward=channels[-2])
        self.token_fn_32 = TokenFN(channel_in=channels[-3], channel_out=channels[-4], dim_feedforward=channels[-3])
        self.token_fn_21 = TokenFN(channel_in=channels[-4], channel_out=channels[-5], dim_feedforward=channels[-4])

        self.upconv5 = UpConv(ch_in=channels[4], ch_out=channels[3], normalization=normalization, activation=activation,
                              num_groups=num_groups)
        self.upconv4 = UpConv(ch_in=channels[3], ch_out=channels[2], normalization=normalization, activation=activation,
                              num_groups=num_groups)
        self.upconv3 = UpConv(ch_in=channels[2], ch_out=channels[1], normalization=normalization, activation=activation,
                              num_groups=num_groups)
        self.upconv2 = UpConv(ch_in=channels[1], ch_out=channels[0], normalization=normalization, activation=activation,
                              num_groups=num_groups)

        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.double_conv_4 = DoubleConv3x3(channels[3] * 2, channels[3], normalization=normalization,
                                           activation=activation, num_groups=num_groups)
        self.double_conv_3 = DoubleConv3x3(channels[2] * 2, channels[2], normalization=normalization,
                                           activation=activation, num_groups=num_groups)
        self.double_conv_2 = DoubleConv3x3(channels[1] * 2, channels[1], normalization=normalization,
                                           activation=activation, num_groups=num_groups)
        self.double_conv_1 = DoubleConv3x3(channels[0] * 2, channels[0], normalization=normalization,
                                           activation=activation, num_groups=num_groups)

    def forward(self, enc_out):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = enc_out
        n, c1, h1, w1 = enc_out_1.shape
        n, c2, h2, w2 = enc_out_2.shape
        n, c3, h3, w3 = enc_out_3.shape
        n, c4, h4, w4 = enc_out_4.shape
        n, c5, h5, w5 = enc_out_5.shape

        input_token_embedding_5 = self.token_embedding_5.weight.unsqueeze(1).repeat(1, n, 1)

        key_5 = value_5 = enc_out_5.flatten(2).permute(2, 0, 1)
        transformer_out_5, token_out_5, attention_map_5 = self.transformer5(
            query=input_token_embedding_5,
            key=key_5,
            value=value_5)

        transformer_out_5 = transformer_out_5.permute(1, 2, 0).view(*enc_out_5.shape)
        src_cat_4 = torch.cat([enc_out_4, self.upconv5(transformer_out_5)], dim=1)
        src_4 = self.double_conv_4(src_cat_4)
        key_4 = value_4 = src_4.flatten(2).permute(2, 0, 1)
        transformer_out_4, token_out_4, attention_map_4 = self.transformer4(
            query=self.token_fn_54(token_out_5),
            key=key_4,
            value=value_4)

        transformer_out_4 = transformer_out_4.permute(1, 2, 0).view(*enc_out_4.shape)
        src_cat_3 = torch.cat([enc_out_3, self.upconv4(transformer_out_4)], dim=1)
        src_3 = self.double_conv_3(src_cat_3)
        key_3 = value_3 = src_3.flatten(2).permute(2, 0, 1)
        transformer_out_3, token_out_3, attention_map_3 = self.transformer3(
            query=self.token_fn_43(token_out_4),
            key=key_3,
            value=value_3)

        transformer_out_3 = transformer_out_3.permute(1, 2, 0).view(*enc_out_3.shape)
        src_cat_2 = torch.cat([enc_out_2, self.upconv3(transformer_out_3)], dim=1)
        src_2 = self.double_conv_2(src_cat_2)
        key_2 = value_2 = src_2.flatten(2).permute(2, 0, 1)
        transformer_out_2, token_out_2, attention_map_2 = self.transformer2(
            query=self.token_fn_32(token_out_3),
            key=key_2,
            value=value_2)

        transformer_out_2 = transformer_out_2.permute(1, 2, 0).view(*enc_out_2.shape)
        src_cat_1 = torch.cat([enc_out_1, self.upconv2(transformer_out_2)], dim=1)
        src_1 = self.double_conv_1(src_cat_1)
        key_1 = value_1 = src_1.flatten(2).permute(2, 0, 1)
        transformer_out_1, token_out_1, attention_map_1 = self.transformer1(
            query=self.token_fn_21(token_out_2),
            key=key_1,
            value=value_1)

        attention_map_5 = attention_map_5.permute(0, 2, 1).view(n, -1, h5, w5)
        attention_map_4 = attention_map_4.permute(0, 2, 1).view(n, -1, h4, w4)
        attention_map_3 = attention_map_3.permute(0, 2, 1).view(n, -1, h3, w3)
        attention_map_2 = attention_map_2.permute(0, 2, 1).view(n, -1, h2, w2)
        attention_map_1 = attention_map_1.permute(0, 2, 1).view(n, -1, h1, w1)

        attention_map_out_5 = torch.clamp(self.upscore5(attention_map_5), min=0, max=1)
        attention_map_out_4 = torch.clamp(self.upscore4(attention_map_4), min=0, max=1)
        attention_map_out_3 = torch.clamp(self.upscore3(attention_map_3), min=0, max=1)
        attention_map_out_2 = torch.clamp(self.upscore2(attention_map_2), min=0, max=1)
        attention_map_out_1 = torch.clamp(attention_map_1, min=0, max=1)
        attention_map_out = [attention_map_out_1,
                             attention_map_out_2,
                             attention_map_out_3,
                             attention_map_out_4,
                             attention_map_out_5]

        return attention_map_out


class SegTransformer(nn.Module):
    def __init__(self, n_channel, start_channel, n_class, input_dim=(512, 512),
                 nhead=4, normalization='bn', activation='relu', num_groups=None, with_background=True):
        super(SegTransformer, self).__init__()
        if input_dim is None:
            input_dim = [512, 512]
        self.n_channel = n_channel
        self.start_channel = start_channel
        self.n_class = n_class

        self.backbone_encoder = Encoder(n_channel=n_channel, start_channel=start_channel,
                                        normalization=normalization, activation=activation, num_groups=num_groups)
        self.segtransformer_decoder = SegTransformerDecoder(start_channel=start_channel, n_class=n_class,
                                                            input_dim=input_dim, nhead=nhead,
                                                            normalization=normalization, activation=activation,
                                                            num_groups=num_groups,
                                                            with_background=with_background)

    def forward(self, x):
        enc_out = self.backbone_encoder(x)
        attention_map_out = self.segtransformer_decoder(enc_out)

        return attention_map_out
