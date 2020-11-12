import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def get_normalization(ch_out, normalization, num_groups=None, ln_norm_shape=None):
    if normalization == 'bn':
        norm = nn.BatchNorm2d(num_features=ch_out)
    elif normalization == 'in':
        norm = nn.InstanceNorm2d(num_features=ch_out)
    elif normalization == 'ln':
        norm = nn.LayerNorm(ln_norm_shape)
    elif normalization == 'gn':
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=ch_out)
    elif normalization is None:
        norm = None
    else:
        raise ValueError(f"No valid activation {normalization} -> Conv")
    return norm


def get_activation(activation):
    if activation == 'relu':
        act = nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        act = nn.LeakyReLU(inplace=False)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'softmax':
        act = nn.Softmax(dim=1)
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation is None:
        return None
    else:
        raise ValueError(f"No valid activation {activation} -> Conv")
    return act


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True,
                 normalization: Optional[str] = 'bn', activation: Optional[str] = 'relu',
                 num_groups: Optional[int] = None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = get_normalization(ch_out, normalization, num_groups=num_groups)
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, deconv_flag=True, normalization='bn', activation='relu', num_groups=None):
        super(UpConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deconv_flag = deconv_flag

        if deconv_flag:
            self.conv = nn.ConvTranspose2d(ch_in, ch_out,
                                           kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
            )
        self.norm = get_normalization(ch_out, normalization, num_groups=num_groups)
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DownConv(nn.Module):
    def __init__(self, ch_in, ch_out, normalization='bn', activation='relu', num_groups=None):
        super(DownConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.conv = Conv(ch_in, ch_out, normalization=normalization, activation=activation, num_groups=num_groups)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class DoubleConv3x3(nn.Module):
    """
        double conv block
    """

    def __init__(self, ch_in, ch_out, normalization='bn', activation='relu', num_groups=None):
        super(DoubleConv3x3, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.module = nn.Sequential(
            Conv(ch_in, ch_out, normalization=normalization, activation=activation, num_groups=num_groups),
            Conv(ch_out, ch_out, normalization=normalization, activation=activation, num_groups=num_groups)
        )

    def forward(self, x):
        out = self.module(x)
        return out


class Encoder(nn.Module):
    def __init__(self, n_channel, start_channel=32, normalization='bn', activation='relu', num_groups=None):
        """
            initialize encoder
        :param n_channel:
        :param start_channel:
        """

        super(Encoder, self).__init__()
        self.input_channel = n_channel
        self.start_chanel = start_channel
        self.channels = np.asarray([start_channel * 2 ** i for i in range(5)])

        self.double_conv_1 = DoubleConv3x3(ch_in=self.input_channel, ch_out=self.channels[0],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_2 = DoubleConv3x3(ch_in=self.channels[0], ch_out=self.channels[1],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_3 = DoubleConv3x3(ch_in=self.channels[1], ch_out=self.channels[2],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_4 = DoubleConv3x3(ch_in=self.channels[2], ch_out=self.channels[3],
                                           normalization=normalization, activation=activation, num_groups=num_groups)
        self.double_conv_5 = DoubleConv3x3(ch_in=self.channels[3], ch_out=self.channels[4],
                                           normalization=normalization, activation=activation, num_groups=num_groups)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        enc_out_1 = self.double_conv_1(x)
        enc_out_2 = self.double_conv_2(self.maxpool_1(enc_out_1))
        enc_out_3 = self.double_conv_3(self.maxpool_2(enc_out_2))
        enc_out_4 = self.double_conv_4(self.maxpool_3(enc_out_3))
        enc_out_5 = self.double_conv_5(self.maxpool_4(enc_out_4))

        return [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5]


class Decoder(nn.Module):
    def __init__(self, start_channel, n_class, deconv_flag=False, normalization='bn', activation='relu', num_groups=None, deep_supervision=False):
        """
            initialize decoder
        :param start_channel:
        :param n_class:
        """
        super(Decoder, self).__init__()
        self.start_channel = start_channel
        self.n_class = n_class
        self.deep_supervision = deep_supervision

        channels = np.asarray([start_channel * 2 ** i for i in range(5)])

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

        # output branch
        if deep_supervision:
            self.output_conv_5 = Conv(ch_in=channels[4], ch_out=n_class, normalization=None, activation=None,
                                      num_groups=None)
            self.output_conv_4 = Conv(ch_in=channels[3], ch_out=n_class, normalization=None, activation=None,
                                      num_groups=None)
            self.output_conv_3 = Conv(ch_in=channels[2], ch_out=n_class, normalization=None, activation=None,
                                      num_groups=None)
            self.output_conv_2 = Conv(ch_in=channels[1], ch_out=n_class, normalization=None, activation=None,
                                      num_groups=None)
        self.output_conv_1 = Conv(ch_in=channels[0], ch_out=n_class, normalization=None, activation=None,
                                  num_groups=None)

    def forward(self, enc_out):
        [enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5] = enc_out

        # decoder
        dec_in_4 = torch.cat([enc_out_4, self.up_conv_5(enc_out_5)], dim=1)
        dec_out_4 = self.double_conv_4(dec_in_4)

        dec_in_3 = torch.cat([enc_out_3, self.up_conv_4(dec_out_4)], dim=1)
        dec_out_3 = self.double_conv_3(dec_in_3)

        dec_in_2 = torch.cat([enc_out_2, self.up_conv_3(dec_out_3)], dim=1)
        dec_out_2 = self.double_conv_2(dec_in_2)

        dec_in_1 = torch.cat([enc_out_1, self.up_conv_2(dec_out_2)], dim=1)
        dec_out_1 = self.double_conv_1(dec_in_1)

        # output
        if self.deep_supervision:
            out_5 = self.output_conv_5(enc_out_5)
            out_4 = self.output_conv_4(dec_out_4)
            out_3 = self.output_conv_3(dec_out_3)
            out_2 = self.output_conv_2(dec_out_2)
            out_1 = self.output_conv_1(dec_out_1)
            return [out_1, out_2, out_3, out_4, out_5]
        else:
            out_1 = self.output_conv_1(dec_out_1)
            return [out_1]


class Unet(nn.Module):
    def __init__(self, n_channel, start_channel, n_class, deep_supervision=False):
        super(Unet, self).__init__()
        self.encoder = Encoder(n_channel=n_channel, start_channel=start_channel)
        self.decoder = Decoder(start_channel=start_channel, n_class=n_class, deep_supervision=deep_supervision)

    def forward(self, x):
        """
        return list of encoder output[(N, C_n, H_n, W_n)]
        """
        enc_out = self.encoder(x)

        """
        return list of decoder output given deep_supervided. 
        if deep_supervised is True, return output from all decoder layers, 
        otherwise, only return the output from the top layer.
        """
        out = self.decoder(enc_out)

        return out
