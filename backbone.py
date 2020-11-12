import torch
import numpy as np
import argparse
import datetime
import os
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import glob

###################################################################################
# set cuda visible
###################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
from data.dataloader2d import create_loader_2d
from data.dataloader3d import create_loader_3d
from models.backbone import Unet
from models.loss import Criterion
from util.yaml_util import load_config_yaml
from util.logging_util import create_logger
from util.model_util import save_checkpoint, load_checkpoint, load_checkpoint_encoder, load_checkpoint_decoder, freeze, \
    unfreeze
from val_unet import val


def main():
    parser = argparse.ArgumentParser(description='Unet training')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config_yaml(args.config)
    data_config = load_config_yaml(config['data_config'])

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M")
    os.makedirs(config['logging_dir'], exist_ok=True)
    logging_path = os.path.join(config['logging_dir'], f'logging_train_{date_time}.txt')
    logger = create_logger(logging_path, stdout=False)

    ###################################################################################
    # construct net
    ###################################################################################
    n_channel = data_config['dataset']['2d']['n_slice']
    n_class = len(data_config['dataset']['3d']['roi_names'])
    if data_config['dataset']['3d']['with_issue_air_mask']:
        n_class += 2
    start_channel = int(config['start_channel'])
    logger.info(f'create model with n_channel={n_channel}, start_channel={start_channel}, n_class={n_class}')

    model = Unet(n_channel=n_channel, start_channel=start_channel, n_class=n_class,
                 deep_supervision=config["deep_supervision"]).to(device)

    logger.info(f"model_dir: {config['ckpt_dir']}")

    ###################################################################################
    # criterion, optimizer, scheduler
    ###################################################################################
    criterion = Criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'])
    if config['deep_supervision']:
        logger.info('Train model using deep supervision')
    else:
        logger.info('Train model using deep supervision')

    ###################################################################################
    # SummaryWriter
    ###################################################################################
    logger.info("Creating writer")
    writer = SummaryWriter(comment=f"LR_{config['lr']}_BS_{config['n_epoch']}")

    ###################################################################################
    # train setup
    ###################################################################################
    global_step = 0
    best_loss = np.inf
    epoch_start = 0

    ###################################################################################
    # load previous model
    ###################################################################################
    if config['load_checkpoint']:
        logger.info(f'Loading model from {os.path.join(config["ckpt_dir"], config["ckpt_fn"])}...')
        model, optimizer, scheduler, epoch_start, global_step = load_checkpoint(model, optimizer, scheduler,
                                                                                config['ckpt_dir'], config['ckpt_fn'],
                                                                                device)
    elif config['load_checkpoint_encoder']:
        logger.info(f'Loading encoder from {os.path.join(config["ckpt_dir"], config["ckpt_fn"])}...')
        model.encoder = load_checkpoint_encoder(model.encoder,
                                                ckpt_dir=config['ckpt_dir'],
                                                ckpt_fn=config['ckpt_fn'],
                                                device=device)
        if config['freeze_encoder']:
            logger.info('Freeze encoder')
            freeze(model.encoder)
    elif config['load_checkpoint_decoder']:
        logger.info(f'Loading decoder from {os.path.join(config["ckpt_dir"], config["ckpt_fn"])}...')
        model.decoder = load_checkpoint_decoder(model.decoder,
                                                ckpt_dir=config['ckpt_dir'],
                                                ckpt_fn=config['ckpt_fn'],
                                                device=device)
        if config['freeze_decoder']:
            logger.info('Freeze decoder')
            freeze(model.decoder)

    ###################################################################################
    # parallel model and data
    ###################################################################################
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    ###################################################################################
    # Dataset
    ###################################################################################
    dataloader_3d = create_loader_3d(data_config, 'train')
    ###################################################################################
    # train
    ###################################################################################
    logger.info(f'Starting training from epoch: {epoch_start}')
    for epoch in range(epoch_start, config['n_epoch']):
        logger.info(f"Epoch: {epoch}/{config['n_epoch']}")
        epoch_loss = 0
        epoch_loss_focal = 0
        epoch_loss_dice = 0
        n_batch_3d = len(dataloader_3d)
        with tqdm(total=n_batch_3d, desc=f"Epoch {epoch + 1}/{config['n_epoch']}", unit='batch') as pbar:
            for batch_3d in dataloader_3d:
                dataloader_2d = create_loader_2d(batch_3d, data_config, 'train')
                n_batch_2d = len(dataloader_2d)
                for idx, batch_2d in enumerate(dataloader_2d):
                    img = batch_2d['img'].to(device=device, dtype=torch.float32)  # [N, n_channel, H, W]
                    mask_gt = batch_2d['mask'].to(device=device, dtype=torch.float32)  # [N, H, W]
                    mask_pred = model(img)
                    mask_flag = batch_2d['mask_flag'].to(device=device, dtype=torch.float32)

                    loss, loss_dict = criterion(pred=mask_pred, target=mask_gt, target_roi_weight=mask_flag)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
                    optimizer.step()

                    global_step += 1
                    loss_scalar = loss_dict["loss"]
                    loss_focal_scalar = loss_dict["focal_loss"]
                    loss_dice_scalar = loss_dict["dice_loss"]
                    epoch_loss += loss_scalar
                    epoch_loss_focal += loss_focal_scalar
                    epoch_loss_dice += loss_dice_scalar

                    pbar.set_postfix(
                        **{'loss (batch)': loss_scalar, 'loss_focal': loss_focal_scalar, 'loss_dice': loss_dice_scalar,
                           'global_step': global_step})

                    if (global_step + 1) % (config['write_summary_loss_batch_step']) == 0:
                        logger.info(
                            f"\tBatch: {idx}/{n_batch_2d}, Loss: {loss_scalar}, Focal_loss: {loss_focal_scalar}, Dice_loss: {loss_dice_scalar}")
                        writer.add_scalar('Loss/train', loss_scalar, global_step)
                        writer.add_scalar('Loss/train_focal', loss_focal_scalar, global_step)
                        writer.add_scalar('Loss/train_dice', loss_dice_scalar, global_step)
                    if (global_step + 1) % (config['write_summary_2d_batch_step']) == 0:
                        writer.add_images('train/images', torch.unsqueeze(img[:, n_channel // 2], 1), global_step)
                        writer.add_images('train/gt_masks', torch.sum(mask_gt, dim=1, keepdim=True), global_step)
                        writer.add_images('train/pred_masks',
                                          torch.sum(mask_pred[0] > 0, dim=1, keepdim=True) >= 1, global_step)
                        writer.add_images('train/pred_masks_raw',
                                          torch.sum(mask_pred[0], dim=1, keepdim=True), global_step)
                pbar.update()

            scheduler.step()
            # log epoch loss
            if (epoch + 1) % config['logging_epoch_step'] == 0:
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Loss_epoch/train', epoch_loss, epoch)
                writer.add_scalar('Loss_epoch/train_focal', epoch_loss_focal, epoch)
                writer.add_scalar('Loss_epoch/train_dice', epoch_loss_dice, epoch)
                logger.info(
                    f"Epoch: {epoch}/{config['n_epoch']}, Train Loss: {epoch_loss}, Train Loss BCE: {epoch_loss_focal}, Train Loss DSC: {epoch_loss_dice}")

            # validation and save model
            if (epoch + 1) % config['val_model_epoch_step'] == 0:
                val_loss, val_focal_loss, val_dice_loss = val(model, criterion, data_config, n_channel, logger, writer,
                                                              global_step, device)
                writer.add_scalar('Loss_epoch/val', val_loss, epoch)
                writer.add_scalar('Loss_epoch/val_focal', val_focal_loss, epoch)
                writer.add_scalar('Loss_epoch/val_dice', val_dice_loss, epoch)
                logger.info(
                    f"Epoch: {epoch}/{config['n_epoch']}, Validation Loss: {val_loss}, Validation Loss Focal: {val_focal_loss}, Validation Loss Dice: {val_dice_loss}")

                os.makedirs(config['ckpt_dir'], exist_ok=True)
                save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                epoch=epoch, global_step=global_step,
                                ckpt_dir=config['ckpt_dir'], ckpt_fn=f'ckpt_{date_time}_Epoch_{epoch}.ckpt')

                if best_loss > val_loss:
                    best_loss = val_loss
                    for filename in glob.glob(os.path.join(config['ckpt_dir'], "best_ckpt*")):
                        os.remove(filename)
                    save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                    epoch=epoch, global_step=global_step,
                                    ckpt_dir=config['ckpt_dir'], ckpt_fn=f'best_ckpt_{date_time}_epoch_{epoch}.ckpt')

        if config['freeze_encoder'] and config['unfreeze_encoder_epoch'] is not None:
            if epoch >= int(config['unfreeze_encoder_epoch']):
                unfreeze(model.module.encoder)
                config['unfreeze_encoder_epoch'] = None
                logger.info(f'Unfreeze encoder at {epoch}')
        if config['freeze_decoder'] and config['unfreeze_decoder_epoch'] is not None:
            if epoch >= int(config['unfreeze_decoder_epoch']):
                unfreeze(model.module.decoder)
                config['unfreeze_decoder_epoch'] = None
                logger.info(f'Unfreeze decoder at {epoch}')
    writer.close()


if __name__ == '__main__':
    main()
