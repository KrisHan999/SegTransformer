import numpy as np
import os
import nrrd
import torch
from torch.utils.data import Dataset
import random

from util.csv_util import read_csv
from data.augmentation import Transformer


class Dataset3d(Dataset):
    def __init__(self, dataset3d_config, phase, random_seed=0):

        random.seed(random_seed)

        self.phase = phase
        self.data_dir = dataset3d_config['data_dir']
        self.roi_names = dataset3d_config['roi_names']
        self.roi_names_dict = dataset3d_config['roi_names_dict']
        self.with_issue_air_mask = dataset3d_config['with_issue_air_mask']
        self.with_background = dataset3d_config['with_background']
        self.pids_path = dataset3d_config['pids_path']
        self.train_pids_path = dataset3d_config['train_pids_path']
        self.test_pids_path = dataset3d_config['test_pids_path']
        self.transformer_config = dataset3d_config['transformer'][phase]
        self.transformer = Transformer(self.transformer_config).play_transform()

        self.pids, self.train_pids, self.test_pids = self.parse_pids()

        if phase == 'train':
            self.used_pids = self.train_pids
        else:
            self.used_pids = self.test_pids

    def parse_pids(self):
        pids = [row[0] for row in read_csv(self.pids_path)]
        train_pids = [row[0] for row in read_csv(self.train_pids_path)]
        testpids = [row[0] for row in read_csv(self.test_pids_path)]
        return pids, train_pids, testpids

    def __len__(self):
        return len(self.used_pids)

    def __getitem__(self, idx):
        self.pid = self.used_pids[idx]
        print('Processing {}'.format(self.pid))
        # load the raw img
        self.img, _ = nrrd.read(os.path.join(self.data_dir, '%s_img.nrrd' % self.pid))
        # load the mask and output the rough mask weight
        self.mask_array, self.mask_flag = self._load_mask(self.pid, self.img.shape)
        # whether mask for each roi is valid or not.

        data = {'pid': self.pid,
                'img': self.img,
                'mask': self.mask_array,
                'mask_flag': self.mask_flag
                }

        data = self.transformer(data)

        if self.with_issue_air_mask:
            # Because the borderMode for air mask is different from other mask
            data['mask'][-1] = torch.ones_like(data['mask'][-1], dtype=torch.uint8) - torch.sum(data['mask'][:-1], dim=0, keepdim=True).gt(0).type(torch.uint8)
        if self.with_background:
            data['mask'][-1] = torch.ones_like(data['mask'][-1], dtype=torch.uint8) - torch.sum(data['mask'][:-1], dim=0, keepdim=True).gt(0).type(torch.uint8)
        # mask -> [N, D, H, W], N is number of rois
        # mask_flag -> [N], N is number of rois, 1 -> roi with mask, 0 -> roi without mask

        # print(f"Extract 3d: {self.pid}: {self.img.shape}, {torch.max(data['label'])} -> 3d datatset")
        return data

    def _load_mask(self, pid, img_shape):
        mask_array = []
        mask_flag = []
        for j, roi in enumerate(self.roi_names):
            if os.path.isfile(os.path.join(self.data_dir, '%s_%s.nrrd' % (pid, roi))):
                m, _ = nrrd.read(os.path.join(self.data_dir, '%s_%s.nrrd' % (pid, roi)))
                m = (m > 0.5).astype(np.uint8)
                mask_array.append(m)
                mask_flag.append(True)
            else:
                print(f"{pid} doesn't have mask for {roi}")
                mask_array.append(np.zeros(img_shape))
                mask_flag.append(False)
        mask_array = np.asarray(mask_array)
        return mask_array, mask_flag





