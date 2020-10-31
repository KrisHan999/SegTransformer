import torch
import torch.nn as nn
import numpy as np

from models.position_encoding import PositionEmbeddingSine2d
from models.backbone import Encoder, Decoder
from models.trasnformer import Transformer


class TokenFN(nn.Module):
    """
    Token feed-forward network. From deep layer to top layer
    """

    def __init__(self, channel_in, channel_out, dim_feedforward, dropout=0.1):
        super(TokenFN, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(channel_in, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, channel_out),
            nn.Dropout(dropout),
            nn.LayerNorm(channel_out))

    def forward(self, token):
        """

        :param token: [L, N, channel_in]
        :return:
            token: [L, N, channel_out]
        """
        token = self.module(token)
        return token


class SegTransformer(nn.Module):
    def __init__(self, n_channel, start_channel, n_class, d_pos=64, input_dim=(512, 512), deep_supervision=False):
        super(SegTransformer, self).__init__()
        if input_dim is None:
            input_dim = [512, 512]
        self.n_channel = n_channel
        self.start_channel = start_channel
        self.n_class = n_class
        self.deep_supervision = deep_supervision

        channels = np.asarray([start_channel * 2 ** i for i in range(5)])

        self.backbone_encoder = Encoder(n_channel=n_channel, start_channel=start_channel)
        self.backbone_decoder = Decoder(start_channel=start_channel, n_class=n_class, deep_supervision=deep_supervision)

        # set this token embedding as the query of TokenTransformer for the deepest layer
        self.token_embedding = nn.Embedding(num_embeddings=n_class, embedding_dim=channels[-1])
        # get position embedding
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
                                        dim_feedforward=256)

        self.token_fn_54 = TokenFN(channel_in=channels[-1], channel_out=channels[-2], dim_feedforward=256)
        self.token_fn_43 = TokenFN(channel_in=channels[-2], channel_out=channels[-3], dim_feedforward=256)
        self.token_fn_32 = TokenFN(channel_in=channels[-3], channel_out=channels[-4], dim_feedforward=256)
        self.token_fn_21 = TokenFN(channel_in=channels[-4], channel_out=channels[-5], dim_feedforward=256)

        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        enc_out = self.backbone_encoder(x)
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = enc_out
        n, c1, h1, w1 = enc_out_1.shape
        n, c2, h2, w2 = enc_out_2.shape
        n, c3, h3, w3 = enc_out_3.shape
        n, c4, h4, w4 = enc_out_4.shape
        n, c5, h5, w5 = enc_out_5.shape

        pos_embedding_1 = self.pos_embedding_1.repeat(1, n, 1).to(x.device)
        pos_embedding_2 = self.pos_embedding_2.repeat(1, n, 1).to(x.device)
        pos_embedding_3 = self.pos_embedding_3.repeat(1, n, 1).to(x.device)
        pos_embedding_4 = self.pos_embedding_4.repeat(1, n, 1).to(x.device)
        pos_embedding_5 = self.pos_embedding_5.repeat(1, n, 1).to(x.device)

        transformer_out_5, token_out_5, token_pos_5, attention_map_5 = self.transformer5(
            src=enc_out_5.flatten(2).permute(2, 0, 1),
            pos_src=pos_embedding_5,
            query=self.token_embedding.weight.unsqueeze(1).repeat(1, B, 1),
            query_pos=None)
        transformer_out_4, token_out_4, token_pos_4, attention_map_4 = self.transformer4(
            src=enc_out_4.flatten(2).permute(2, 0, 1),
            pos_src=pos_embedding_4,
            query=self.token_fn_54(token_out_5),
            query_pos=token_pos_5)
        transformer_out_3, token_out_3, token_pos_3, attention_map_3 = self.transformer3(
            src=enc_out_3.flatten(2).permute(2, 0, 1),
            pos_src=pos_embedding_3,
            query=self.token_fn_43(token_out_4),
            query_pos=token_pos_4)
        transformer_out_2, token_out_2, token_pos_2, attention_map_2 = self.transformer2(
            src=enc_out_2.flatten(2).permute(2, 0, 1),
            pos_src=pos_embedding_2,
            query=self.token_fn_32(token_out_3),
            query_pos=token_pos_3)
        transformer_out_1, token_out_1, token_pos_1, attention_map_1 = self.transformer1(
            src=enc_out_1.flatten(2).permute(2, 0, 1),
            pos_src=pos_embedding_1,
            query=self.token_fn_21(token_out_2),
            query_pos=token_pos_2)
        transformer_out_5 = transformer_out_5.permute(1, 2, 0).view(*enc_out_5.shape)
        transformer_out_4 = transformer_out_4.permute(1, 2, 0).view(*enc_out_4.shape)
        transformer_out_3 = transformer_out_3.permute(1, 2, 0).view(*enc_out_3.shape)
        transformer_out_2 = transformer_out_2.permute(1, 2, 0).view(*enc_out_2.shape)
        transformer_out_1 = transformer_out_1.permute(1, 2, 0).view(*enc_out_1.shape)

        transformer_out = [transformer_out_1,
                           transformer_out_2,
                           transformer_out_3,
                           transformer_out_4,
                           transformer_out_5
                           ]

        out = self.backbone_decoder(transformer_out)

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

        return out, attention_map_out
