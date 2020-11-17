import torch
import torch.nn as nn
from models.backbone import DoubleConv3x3, Encoder, UpConv, Conv
import numpy as np
from models.GSegTrans.transformer import Transformer
from models.localAttn import LocalAttnTransformer


class Backbone(nn.Module):
    '''
        Add transformer to the end of last unet decoder layer
    '''

    def __init__(self, n_class, n_channel, start_channel=32, nhead=4, deconv_flag=True, normalization='bn',
                 activation='relu', num_groups=None, with_background=False):
        super(Backbone, self).__init__()
        channels = np.asarray([start_channel * 2 ** i for i in range(5)])
        self.encoder = Encoder(n_channel=n_channel, start_channel=start_channel, normalization=normalization,
                               activation=activation, num_groups=num_groups)
        # decoder
        self.up_conv_5 = UpConv(ch_in=channels[4], ch_out=channels[3], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_4 = UpConv(ch_in=channels[3], ch_out=channels[2], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_3 = UpConv(ch_in=channels[2], ch_out=channels[1], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_2 = UpConv(ch_in=channels[1], ch_out=channels[0], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)

        self.double_conv_4 = DoubleConv3x3(ch_in=channels[4], ch_out=channels[3],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_3 = DoubleConv3x3(ch_in=channels[3], ch_out=channels[2],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_2 = DoubleConv3x3(ch_in=channels[2], ch_out=channels[1],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_1 = DoubleConv3x3(ch_in=channels[1], ch_out=channels[0],
                                           normalization=normalization, activation=activation, num_groups=num_groups)

        n_decode_channel = channels[0]
        self.decoder_transformer = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                               nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                               d_model=n_decode_channel,
                                               dim_feedforward=64,
                                               with_background=with_background)
        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=n_decode_channel)

    def forward(self, x):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = self.encoder(x)

        # decoder
        dec_in_4 = torch.cat([enc_out_4, self.up_conv_5(enc_out_5)], dim=1)
        dec_out_4 = self.double_conv_4(dec_in_4)

        dec_in_3 = torch.cat([enc_out_3, self.up_conv_4(dec_out_4)], dim=1)
        dec_out_3 = self.double_conv_3(dec_in_3)

        dec_in_2 = torch.cat([enc_out_2, self.up_conv_3(dec_out_3)], dim=1)
        dec_out_2 = self.double_conv_2(dec_in_2)

        dec_in_1 = torch.cat([enc_out_1, self.up_conv_2(dec_out_2)], dim=1)
        dec_out_1 = self.double_conv_1(dec_in_1)

        n, c, h, w = dec_out_1.shape

        input_token_embedding = self.token_embedding.weight.unsqueeze(1).repeat(1, n, 1)
        src = dec_out_1.flatten(2).permute(2, 0, 1)
        _, _, attention_map = self.decoder_transformer(key=src, value=src, query=input_token_embedding)
        attention_map = attention_map.permute(0, 2, 1).view(n, -1, h, w)
        attention_map = torch.clamp(attention_map, min=0, max=1)
        return [attention_map]


class Backbone_2(nn.Module):
    '''
        Fuse all unet encoder feature, single transformer branch.
    '''

    def __init__(self, n_class, n_channel, start_channel=32, n_decode_channel=128, nhead=4,
                 normalization='bn', activation='relu', num_groups=None, with_background=False):
        super(Backbone_2, self).__init__()
        self.channels = np.asarray([start_channel * 2 ** i for i in range(5)])
        self.encoder = Encoder(n_channel=n_channel, start_channel=start_channel, normalization=normalization,
                               activation=activation, num_groups=num_groups)
        self.upsample5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fusion_conv = DoubleConv3x3(ch_in=self.channels.sum(), ch_out=n_decode_channel,
                                         normalization=normalization, activation=activation, num_groups=num_groups)
        self.decoder_transformer = Transformer(num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                               nhead_encoder=nhead, nhead_tokener=nhead, nhead_decoder=nhead,
                                               d_model=n_decode_channel,
                                               dim_feedforward=512,
                                               with_background=with_background)

        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=n_decode_channel)

    def forward(self, x):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = self.encoder(x)
        enc_out_up_1 = enc_out_1
        enc_out_up_2 = self.upsample2(enc_out_2)
        enc_out_up_3 = self.upsample3(enc_out_3)
        enc_out_up_4 = self.upsample4(enc_out_4)
        enc_out_up_5 = self.upsample5(enc_out_5)

        fusion_input = torch.cat([enc_out_up_1,
                                  enc_out_up_2,
                                  enc_out_up_3,
                                  enc_out_up_4,
                                  enc_out_up_5],
                                 dim=1)

        fusion_out = self.fusion_conv(fusion_input)

        n, c, h, w = fusion_out.shape

        input_token_embedding = self.token_embedding.weight.unsqueeze(1).repeat(1, n, 1)
        src = fusion_out.flatten(2).permute(2, 0, 1)
        _, _, attention_map = self.decoder_transformer(key=src, value=src, query=input_token_embedding)
        attention_map = attention_map.permute(0, 2, 1).view(n, -1, h, w)
        attention_map = torch.clamp(attention_map, min=0, max=1)
        return [attention_map]


class Backbone_3(nn.Module):
    '''
        Add transformer to the end of last unet decoder layer
    '''

    def __init__(self, n_class, n_channel, start_channel=32, kernel_size=3, padding=1, nhead=4, deconv_flag=True,
                 normalization='bn',
                 activation='relu', num_groups=None):
        super(Backbone_3, self).__init__()
        channels = np.asarray([start_channel * 2 ** i for i in range(5)])
        self.encoder = Encoder(n_channel=n_channel, start_channel=start_channel, normalization=normalization,
                               activation=activation, num_groups=num_groups)
        # decoder
        self.up_conv_5 = UpConv(ch_in=channels[4], ch_out=channels[3], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_4 = UpConv(ch_in=channels[3], ch_out=channels[2], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_3 = UpConv(ch_in=channels[2], ch_out=channels[1], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)
        self.up_conv_2 = UpConv(ch_in=channels[1], ch_out=channels[0], deconv_flag=deconv_flag,
                                normalization=normalization, activation=activation, num_groups=num_groups)

        self.double_conv_4 = DoubleConv3x3(ch_in=channels[4], ch_out=channels[3],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_3 = DoubleConv3x3(ch_in=channels[3], ch_out=channels[2],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_2 = DoubleConv3x3(ch_in=channels[2], ch_out=channels[1],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_1 = DoubleConv3x3(ch_in=channels[1], ch_out=channels[0],
                                           normalization=normalization, activation=activation, num_groups=num_groups)

        self.decoder_transformer = LocalAttnTransformer(kernel_size=kernel_size, padding=padding,
                                                        num_layers=2, d_model=channels[0], nhead=nhead,
                                                        dim_feedforward=channels[0] * 2)
        self.output_conv = Conv(ch_in=channels[0], ch_out=n_class, normalization=None, activation=None,
                                num_groups=None)

    def forward(self, x):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = self.encoder(x)

        # decoder
        dec_in_4 = torch.cat([enc_out_4, self.up_conv_5(enc_out_5)], dim=1)
        dec_out_4 = self.double_conv_4(dec_in_4)

        dec_in_3 = torch.cat([enc_out_3, self.up_conv_4(dec_out_4)], dim=1)
        dec_out_3 = self.double_conv_3(dec_in_3)

        dec_in_2 = torch.cat([enc_out_2, self.up_conv_3(dec_out_3)], dim=1)
        dec_out_2 = self.double_conv_2(dec_in_2)

        dec_in_1 = torch.cat([enc_out_1, self.up_conv_2(dec_out_2)], dim=1)
        dec_out_1 = self.double_conv_1(dec_in_1)

        local_feature, attention_map = self.decoder_transformer(dec_out_1)
        feature_fusion = dec_out_1 + local_feature
        out = self.output_conv(feature_fusion)
        return [out]
