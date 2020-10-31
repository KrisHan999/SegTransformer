import torch
import numpy as np
import argparse
import os
import nrrd
from tqdm import tqdm

from data.dataloader2d import create_loader_2d
from data.dataloader3d import create_loader_3d
from models.backbone import Unet
from util.yaml_util import load_config_yaml
from util.model_util import load_checkpoint_model


def pred_with_model(model, ckpt_dir, ckpt_fn, pred_save_dir, config, data_config, roi_names, device):
    model = load_checkpoint_model(model, ckpt_dir, ckpt_fn, device)

    ###################################################################################
    # Dataset
    ###################################################################################
    dataloader_3d = create_loader_3d(data_config, 'pred')
    n_batch_3d = len(dataloader_3d)
    with torch.no_grad():
        with tqdm(total=n_batch_3d, desc=f"Predicting", unit='batch') as pbar:
            for batch_3d in dataloader_3d:
                assert len(batch_3d) == 1, 'len(batch_3d) in pred.py must be set to 1.'
                pid = batch_3d[0]['pid']
                mask_pred_all = torch.zeros_like(batch_3d[0]['mask'], dtype=torch.bool).to(device)  # shape -> (N_roi, D, H, W)
                dataloader_2d = create_loader_2d(batch_3d, data_config, 'pred')
                for idx, batch_2d in enumerate(dataloader_2d):
                    img = batch_2d['img'].to(device=device, dtype=torch.float32)  # [N, n_channel, H, W]
                    target_slice = batch_2d['target_slice']  # shape -> [N]
                    mask_pred = model(img)  # shape -> (N, C, H, W), C is N_roi
                    mask_pred = mask_pred[0]
                    mask_pred = torch.transpose(mask_pred, 0, 1)  # shape -> (C, N, H, W)
                    mask_pred_all[:, target_slice] = mask_pred.gt(0)

                os.makedirs(config['pred_save_dir'], exist_ok=True)
                for i, roi in enumerate(roi_names):
                    nrrd.write(os.path.join(pred_save_dir, f'{pid}_{roi}.nrrd'), mask_pred_all[i].numpy().astype(np.uint8))
                pbar.update()


def main():
    parser = argparse.ArgumentParser(description='Predicting...')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config_yaml(args.config)
    data_config = load_config_yaml(config['data_config'])

    ###################################################################################
    # construct net
    ###################################################################################
    roi_names = data_config['dataset']['3d']['roi_names']
    n_channel = data_config['dataset']['2d']['n_slice']
    n_class = len(roi_names)
    start_channel = int(config['start_channel'])
    model = Unet(n_channel=n_channel, start_channel=start_channel, n_class=n_class).to(device)

    ckpt_dir = config['ckpt_dir']
    ckpt_fn = config['ckpt_fn']
    pred_save_dir = config['pred_save_dir']
    pred_with_model(model, ckpt_dir, ckpt_fn, pred_save_dir, config, data_config, roi_names, device)



if __name__ == '__main__':
    main()

