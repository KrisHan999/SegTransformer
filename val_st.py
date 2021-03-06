import torch
from data.dataloader2d import create_loader_2d
from data.dataloader3d import create_loader_3d
from tqdm import tqdm


def val(model, criterion, config, roi_names, data_config, n_channel, logger, writer, global_step, device):
    model.eval()
    dataloader_3d = create_loader_3d(data_config, 'val')
    val_loss = 0
    val_dice_loss = 0
    val_attn_loss_dict = {}
    n_batch_3d = len(dataloader_3d)
    with torch.no_grad():
        with tqdm(total=n_batch_3d, desc="Validation execution.", unit='batch') as pbar:
            for idx_3d, batch_3d in enumerate(dataloader_3d):
                dataloader_2d = create_loader_2d(batch_3d, data_config, 'val')
                for idx, batch_2d in enumerate(dataloader_2d):
                    img = batch_2d['img'].to(device=device, dtype=torch.float32)  # [N, n_channel, H, W]
                    mask_gt = batch_2d['mask'].to(device=device, dtype=torch.float32)  # [N, H, W]
                    mask_flag = batch_2d['mask_flag'].to(device=device, dtype=torch.float32)
                    attention_map_out = model(img)
                    loss_attn_map, loss_dict_attn_map = criterion(pred=attention_map_out, target=mask_gt,
                                                                  target_roi_weight=mask_flag, deep_supervision=True,
                                                                  need_sigmoid=False,
                                                                  layer_weight=config['loss']['attention_loss_weight'],
                                                                  for_val=True)
                    loss = loss_attn_map
                    loss_scalar = loss.detach().item()
                    loss_dice_scalar = loss_dict_attn_map['layer_0']["dice_loss"]
                    val_loss += loss_scalar
                    val_dice_loss += loss_dice_scalar

                    for key, value in loss_dict_attn_map.items():
                        val_attn_loss_dict.setdefault(key, dict())
                        val_attn_loss_dict[key].setdefault("epoch_attn_loss_dice", 0)
                        val_attn_loss_dict[key]["epoch_attn_loss_dice"] += value["dice_loss"]

                    pbar.set_postfix(**{'loss (batch)': loss_scalar, 'loss_dice': loss_dice_scalar})
                logger.info(
                    f"\tBatch: {idx_3d}/{n_batch_3d}, Loss: {loss_scalar}, Dice_loss: {loss_dice_scalar}")
                pbar.update()
            writer.add_scalar('Loss_val/val', loss_scalar, global_step)
            writer.add_scalar('Loss_val/val_dice', loss_dice_scalar, global_step)
            for key, value in loss_dict_attn_map.items():
                writer.add_scalar(f'Loss_val/val_dice/attention_{key}', value["dice_loss"], global_step)

            writer.add_images('val/images', torch.unsqueeze(img[:, n_channel // 2], 1), global_step)

            for r_i, roi_name in enumerate(roi_names):
                writer.add_images(f'val/masks_{roi_name}_gt', mask_gt[:, r_i:r_i + 1], global_step)
                writer.add_images(f'val/masks_{roi_name}_pred', attention_map_out[0][:, r_i:r_i + 1], global_step)
            if data_config['dataset']['3d']['with_issue_air_mask']:
                writer.add_images('val/masks_gt', torch.sum(mask_gt[:, :-2], dim=1, keepdim=True),
                                  global_step)
                writer.add_images('val/masks_pred',
                                  torch.sum(attention_map_out[0][:, :-2], dim=1, keepdim=True),
                                  global_step)
            elif data_config['dataset']['3d']['with_background']:
                writer.add_images('val/masks_gt', torch.sum(mask_gt[:, :-1], dim=1, keepdim=True), global_step)
                writer.add_images('val/masks_pred',
                                  torch.sum(attention_map_out[0][:, :-1], dim=1, keepdim=True), global_step)
            else:
                writer.add_images('val/masks_gt', torch.sum(mask_gt, dim=1, keepdim=True), global_step)
                writer.add_images('val/masks_pred',
                                  torch.sum(attention_map_out[0], dim=1, keepdim=True), global_step)
            for l_i, attn_map_single in enumerate(attention_map_out):
                for r_i, roi_name in enumerate(roi_names):
                    writer.add_images(f'val/masks_{roi_name}_pred_layer_{l_i}',
                                      attn_map_single[:, r_i:r_i + 1], global_step)
    model.train()
    return val_loss, val_dice_loss, val_attn_loss_dict
