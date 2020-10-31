import torch.nn as nn
import math
import torch


class PositionEmbeddingSine1d(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """

        :param num_pos_feats: num of features for position encoding
        :param temperature:
        :param normalize: whether normalize the position to [0, 1]
        :param scale: scale for normalized position: [0, 1] -> [0, scale]
        """
        super(PositionEmbeddingSine1d, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normaliza should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, spatial_dims):
        """
        Generate fixed position embedding given spatial dims
        :param spatial_dims: tensor -> [N, D]
        :return:
        """
        mask = torch.ones(*spatial_dims, device=spatial_dims.device)
        embed = mask.cumsum(dim=1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=spatial_dims.device)
        dim_t = 10000 ** ((2 * dim_t // 2) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos.permute(0, 2, 1)
        return pos


class PositionEmbeddingSine2d(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """

        :param num_pos_feats: num of features for position encoding
        :param temperature:
        :param normalize: whether normalize the position to [0, 1]
        :param scale: scale for normalized position: [0, 1] -> [0, scale]
        """
        super(PositionEmbeddingSine2d, self).__init__()
        self.num_pos_feats = num_pos_feats//2   # final output is still the input num_pos_feats,
                                                # because stack y and x positional embedding
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normaliza should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, spatial_dims):
        """
        Generate fixed position embedding given spatial dims
        :param spatial_dims: tensor -> [N, H, W]
        :return: pos: tensor -> [N, num_pos_feats, H, W]
        """
        mask = torch.ones(*spatial_dims, device=spatial_dims.device)
        y_embed = mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=spatial_dims.device)
        dim_t = 10000 ** ((2 * dim_t // 2) / self.num_pos_feats)

        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, None] / dim_t

        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
