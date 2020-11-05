import torch
import torch.nn as nn
import numpy as np

from models.position_encoding import PositionEmbeddingSine2d
from models.backbone import Encoder
from models.trasnformer import Transformer
from models.segtransformer_v1 import TokenFN
from models.backbone import Conv, DoubleConv3x3, UpConv


class SegTransformerDecoder(nn.Module):
    def __init__(self, start_channel, n_class, d_pos=64, input_dim=(512, 512)):
        super(SegTransformerDecoder, self).__init__()
        if input_dim is None:
            input_dim = [512, 512]
        self.start_channel = start_channel
        self.n_class = n_class
        channels = np.asarray([start_channel * 2 ** i for i in range(5)])  # 32, 62, 128, 256, 512

        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=channels[-1])
        # get position embedding
        self.avg_pool_embedding = nn.AvgPool2d(kernel_size=32)  # 32 = 512 / 2**4
        pos_embedding_generater = PositionEmbeddingSine2d(num_pos_feats=d_pos)
        temp_pos_embedding_1 = pos_embedding_generater(torch.tensor([1, *input_dim]))  # [1, d_pos, H, W]
        temp_pos_embedding_2 = temp_pos_embedding_1[:, :, ::2, ::2]
        temp_pos_embedding_3 = temp_pos_embedding_2[:, :, ::2, ::2]
        temp_pos_embedding_4 = temp_pos_embedding_3[:, :, ::2, ::2]
        temp_pos_embedding_5 = temp_pos_embedding_4[:, :, ::2, ::2]
        self.pos_embedding_1 = temp_pos_embedding_1.flatten(2).permute(2, 0, 1)
        self.pos_embedding_2 = temp_pos_embedding_2.flatten(2).permute(2, 0, 1)
        self.pos_embedding_3 = temp_pos_embedding_3.flatten(2).permute(2, 0, 1)
        self.pos_embedding_4 = temp_pos_embedding_4.flatten(2).permute(2, 0, 1)
        self.pos_embedding_5 = temp_pos_embedding_5.flatten(2).permute(2, 0, 1)

        self.transformer5 = Transformer(need_generate_query_pos=True,
                                        num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=4, nhead_tokener=4, nhead_decoder=4,
                                        d_model=channels[-1] + d_pos, d_pos=d_pos,
                                        dim_feedforward=512)

        self.transformer4 = Transformer(need_generate_query_pos=False,
                                        num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=4, nhead_tokener=4, nhead_decoder=4,
                                        d_model=channels[-2] + d_pos, d_pos=d_pos,
                                        dim_feedforward=512)

        self.transformer3 = Transformer(need_generate_query_pos=False,
                                        num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=4, nhead_tokener=4, nhead_decoder=4,
                                        d_model=channels[-3] + d_pos, d_pos=d_pos,
                                        dim_feedforward=512)

        self.transformer2 = Transformer(need_generate_query_pos=False,
                                        num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=4, nhead_tokener=4, nhead_decoder=4,
                                        d_model=channels[-4] + d_pos, d_pos=d_pos,
                                        dim_feedforward=512)

        self.transformer1 = Transformer(need_generate_query_pos=False,
                                        num_layers_encoder=2, num_layers_tokener=2, num_layers_decoder=1,
                                        nhead_encoder=4, nhead_tokener=4, nhead_decoder=4,
                                        d_model=channels[-5] + d_pos, d_pos=d_pos,
                                        dim_feedforward=512)

        self.token_fn_54 = TokenFN(channel_in=channels[-1], channel_out=channels[-2], dim_feedforward=256)
        self.token_fn_43 = TokenFN(channel_in=channels[-2], channel_out=channels[-3], dim_feedforward=256)
        self.token_fn_32 = TokenFN(channel_in=channels[-3], channel_out=channels[-4], dim_feedforward=256)
        self.token_fn_21 = TokenFN(channel_in=channels[-4], channel_out=channels[-5], dim_feedforward=256)

        self.upconv5 = UpConv(ch_in=channels[4], ch_out=channels[3])
        self.upconv4 = UpConv(ch_in=channels[3], ch_out=channels[2])
        self.upconv3 = UpConv(ch_in=channels[2], ch_out=channels[1])
        self.upconv2 = UpConv(ch_in=channels[1], ch_out=channels[0])

        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.double_conv_4 = DoubleConv3x3(channels[3]*2, channels[3])
        self.double_conv_3 = DoubleConv3x3(channels[2] * 2, channels[2])
        self.double_conv_2 = DoubleConv3x3(channels[1] * 2, channels[1])
        self.double_conv_1 = DoubleConv3x3(channels[0] * 2, channels[0])

    def forward(self, enc_out):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = enc_out
        n, c1, h1, w1 = enc_out_1.shape
        n, c2, h2, w2 = enc_out_2.shape
        n, c3, h3, w3 = enc_out_3.shape
        n, c4, h4, w4 = enc_out_4.shape
        n, c5, h5, w5 = enc_out_5.shape

        pos_embedding_1 = self.pos_embedding_1.repeat(1, n, 1).to(enc_out_1.device)
        pos_embedding_2 = self.pos_embedding_2.repeat(1, n, 1).to(enc_out_2.device)
        pos_embedding_3 = self.pos_embedding_3.repeat(1, n, 1).to(enc_out_3.device)
        pos_embedding_4 = self.pos_embedding_4.repeat(1, n, 1).to(enc_out_4.device)
        pos_embedding_5 = self.pos_embedding_5.repeat(1, n, 1).to(enc_out_5.device)

        input_token_embedding = self.token_embedding.weight.unsqueeze(1).repeat(1, n, 1) + self.avg_pool_embedding(
            enc_out_5).squeeze().unsqueeze(0).repeat(self.n_class, 1, 1)

        src_5 = enc_out_5.flatten(2).permute(2, 0, 1)
        transformer_out_5, token_out_5, token_pos_5, attention_map_5 = self.transformer5(
            src=src_5,
            pos_src=pos_embedding_5,
            query=input_token_embedding,
            query_pos=None)

        transformer_out_5 = transformer_out_5.permute(1, 2, 0).view(*enc_out_5.shape) + enc_out_5
        src_cat_4 = torch.cat([enc_out_4, self.upconv5(transformer_out_5)], dim=1)
        src_4 = self.double_conv_4(src_cat_4)
        src_4 = src_4.flatten(2).permute(2, 0, 1)
        transformer_out_4, token_out_4, token_pos_4, attention_map_4 = self.transformer4(
            src=src_4,
            pos_src=pos_embedding_4,
            query=self.token_fn_54(token_out_5),
            query_pos=token_pos_5)

        transformer_out_4 = transformer_out_4.permute(1, 2, 0).view(*enc_out_4.shape) + enc_out_4
        src_cat_3 = torch.cat([enc_out_3, self.upconv4(transformer_out_4)], dim=1)
        src_3 = self.double_conv_3(src_cat_3)
        src_3 = src_3.flatten(2).permute(2, 0, 1)
        transformer_out_3, token_out_3, token_pos_3, attention_map_3 = self.transformer3(
            src=src_3,
            pos_src=pos_embedding_3,
            query=self.token_fn_43(token_out_4),
            query_pos=token_pos_4)

        transformer_out_3 = transformer_out_3.permute(1, 2, 0).view(*enc_out_3.shape) + enc_out_3
        src_cat_2 = torch.cat([enc_out_2, self.upconv3(transformer_out_3)], dim=1)
        src_2 = self.double_conv_2(src_cat_2)
        src_2 = src_2.flatten(2).permute(2, 0, 1)
        transformer_out_2, token_out_2, token_pos_2, attention_map_2 = self.transformer2(
            src=src_2,
            pos_src=pos_embedding_2,
            query=self.token_fn_32(token_out_3),
            query_pos=token_pos_3)

        transformer_out_2 = transformer_out_2.permute(1, 2, 0).view(*enc_out_2.shape) + enc_out_2
        src_cat_1 = torch.cat([enc_out_1, self.upconv2(transformer_out_2)], dim=1)
        src_1 = self.double_conv_1(src_cat_1)
        src_1 = src_1.flatten(2).permute(2, 0, 1)
        transformer_out_1, token_out_1, token_pos_1, attention_map_1 = self.transformer1(
            src=src_1,
            pos_src=pos_embedding_1,
            query=self.token_fn_21(token_out_2),
            query_pos=token_pos_2)

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


class OutputBranch(nn.Module):
    def __init__(self, start_channel, n_class):
        super(OutputBranch, self).__init__()
        initial_channel = start_channel + n_class * 5
        self.module = nn.Sequential(
            DoubleConv3x3(ch_in=initial_channel, ch_out=start_channel),
            Conv(ch_in=start_channel, ch_out=n_class, normalization=None, activation=None,
                 num_groups=None)
        )

    def forward(self, x, attention_map_out):

        x_all = torch.cat([x, torch.cat(attention_map_out, dim=1)], dim=1)
        out = self.module(x_all)
        return out


class SegTransformer_V2(nn.Module):
    def __init__(self, n_channel, start_channel, n_class, d_pos=64, input_dim=(512, 512)):
        super(SegTransformer_V2, self).__init__()
        if input_dim is None:
            input_dim = [512, 512]
        self.n_channel = n_channel
        self.start_channel = start_channel
        self.n_class = n_class

        self.backbone_encoder = Encoder(n_channel=n_channel, start_channel=start_channel)
        self.segtransformer_decoder = SegTransformerDecoder(start_channel=start_channel, n_class=n_class, d_pos=d_pos, input_dim=input_dim)

    def forward(self, x):
        enc_out = self.backbone_encoder(x)
        attention_map_out = self.segtransformer_decoder(enc_out)

        return attention_map_out
