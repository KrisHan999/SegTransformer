import torch
from torch.utils.data import Dataset
import numpy as np


class Dataset2d(Dataset):
    def __init__(self, dataset_config, batch3d, phase):
        super(Dataset2d, self).__init__()
        self.batch_size_3d = len(batch3d)
        self.phase = phase
        self.n_slice = dataset_config['n_slice']
        self.data, self.z_slice_all, self.pid_all = self.parse_batch(batch3d)
        self.z_slice_per_3d_len = None

    def parse_batch(self, batch3d):

        data = dict()
        z_slice_all = []
        pid_all = []
        for item in batch3d:
            data[item['pid']] = dict()
            data[item['pid']]['img'] = item['img']
            data[item['pid']]['mask'] = item['mask']
            data[item['pid']]['mask_flag'] = item['mask_flag']
            # each item should have z_slice with same length
            z_slice = list(range(item['img'].shape[0] - (self.n_slice - 1)))
            data[item['pid']]['z_slcie'] = z_slice
            z_slice_all = z_slice_all + z_slice
            pid_all = pid_all + [item['pid']] * len(z_slice)

        del batch3d
        return data, z_slice_all, pid_all

    def __len__(self):
        return len(self.z_slice_all)

    def __getitem__(self, idx):

        pid = self.pid_all[idx]
        z_slice = self.z_slice_all[idx]
        img_slice_idx_range = slice(z_slice, z_slice + self.n_slice)
        target_slice = z_slice + self.n_slice // 2

        img_patch = self.data[pid]['img'][img_slice_idx_range]
        mask = self.data[pid]['mask'][:, target_slice]
        mask_flag = self.data[pid]['mask_flag']  # 3d volume roi flag

        data = {'pid': pid,
                'img': img_patch,  # [n_slice, H, W]
                'mask': mask,  # [n_rois, H, W]
                'target_slice': target_slice,  # scalar
                'mask_flag': mask_flag  # [C]
                }

        return data
