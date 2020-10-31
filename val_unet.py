import torch
from data.dataloader2d import create_loader_2d
from data.dataloader3d import create_loader_3d
from tqdm import tqdm


def val(model, criterion, data_config, n_channel, logger, writer, global_step, device):
    model.eval()
    dataloader_3d = create_loader_3d(data_config, 'val')
    val_loss = 0
    val_focal_loss = 0
    val_dice_loss = 0
    n_batch_3d = len(dataloader_3d)
    with torch.no_grad():
        with tqdm(total=n_batch_3d, desc="Validation execution.", unit='batch') as pbar:
            for idx_3d, batch_3d in enumerate(dataloader_3d):
                dataloader_2d = create_loader_2d(batch_3d, data_config, 'val')
                for idx, batch_2d in enumerate(dataloader_2d):
                    img = batch_2d['img'].to(device=device, dtype=torch.float32)  # [N, n_channel, H, W]
                    mask_gt = batch_2d['mask'].to(device=device, dtype=torch.float32)  # [N, H, W]
                    mask_flag = batch_2d['mask_flag'].to(device=device, dtype=torch.float32)
                    mask_pred = model(img)
                    loss, loss_dict = criterion(pred=mask_pred, target=mask_gt, target_roi_weight=mask_flag)
                    loss_scalar = loss.item()
                    loss_focal_scalar = loss_dict["focal_loss"]
                    loss_dice_scalar = loss_dict["dice_loss"]
                    val_loss += loss_dict["loss"]
                    val_focal_loss += loss_focal_scalar
                    val_dice_loss += loss_dice_scalar

                    pbar.set_postfix(**{'loss (batch)': loss_scalar,  'loss_focal': loss_focal_scalar, 'loss_dice': loss_dice_scalar})
                logger.info(f"\tBatch: {idx_3d}/{n_batch_3d}, Loss: {loss_scalar}, Focal_loss: {loss_focal_scalar}, Dice_loss: {loss_dice_scalar}")
                pbar.update()
            writer.add_scalar('Loss/val', loss_scalar, global_step)
            writer.add_scalar('Loss/val_focal', loss_focal_scalar, global_step)
            writer.add_scalar('Loss/val_dice', loss_dice_scalar, global_step)
            writer.add_images('val/images', torch.unsqueeze(img[:, n_channel // 2], 1), global_step)
            writer.add_images('val/gt_masks', torch.sum(mask_gt, dim=1, keepdim=True), global_step)

            writer.add_images('val/pred_masks',
                              torch.sum(mask_pred[0] > 0, dim=1, keepdim=True) >= 1, global_step)
            writer.add_images('val/pred_masks_raw',
                              torch.sum(mask_pred[0], dim=1, keepdim=True), global_step)
    model.train()
    return val_loss, val_focal_loss, val_dice_loss