import numpy as np
import os
import matplotlib.pyplot as plt
import ipywidgets as w


def mask_array_to_mask(mask_array, num_rois):
    """
        combine multiple binary masks to one mask. Mask array has no background component, while mask has background as label 0
    """

    num_mask, D, H, W = mask_array.shape
    assert num_mask == num_rois, "mask array doesn't have same number of mask as number of target rois"
    mask = np.zeros((D, H, W))
    for i in range(num_mask):
        mask[mask_array[i] > 0] = i+1
    return mask


def mask_to_mask_array(mask, num_rois):
    """
        generate one mask for each roi. mask has background as label 0, while mask_array has no background label.
    params:
        mask: ndarray that contains mask for multiple rois
        num_rois: number of all target roi.
    """
    D, H, W = mask.shape
    mask_array = np.zeros((num_rois, D, H, W))
    for i in range(num_rois):
        mask_array[i][mask == i + 1] = 1
    return mask_array


def show_3d(volume):
    """
        This function currently can only be used in notebook.
    """
    z_any = [np.any(slice_2d) for slice_2d in volume]
    z_range = np.where(z_any)[0]

    def fz(k):
        plt.figure(figsize=(15, 15))
        print(z_range[k])
        plt.imshow(volume[z_range[k]])
        plt.colorbar()

    w.interact(fz, k=w.IntSlider(min=0, max=len(z_range) - 1, step=1, value=0))


def show_2_3d(volume_1, volume_2):
    def fz(k):
        print(k)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].imshow(volume_1[k])
        axes[1].imshow(volume_2[k])
    w.interact(fz, k=w.IntSlider(min=0, max=len(volume_1) - 1, step=1, value=0))



def show_img_mask(volume_img, volume_mask):
    """
        show image and one mask in same figure.
    params:
        volume_img: shape -> (D, H, W)
        volume_mask: shape -> (D, H, W)
    """

    z_any = [np.any(m) for m in volume_mask]
    z_range = np.where(z_any)[0]

    def fz(k):
        print(z_range[k])
        plt.figure(figsize=(15, 15))
        plt.imshow(volume_img[z_range[k]], cmap='gray', alpha=0.7)
        plt.imshow(volume_mask[z_range[k]], alpha=0.3)

    w.interact(fz, k=w.IntSlider(min=0, max=len(z_range) - 1, step=1, value=0))


def show_all_roi(volume_img, volume_mask_list):
    """
        show img and list of roi mask in one figure. Only show the section where mask is not empty.
    params:
        volume_img: shape -> (D, H, W)
        volume_mask_list: list of volume_mask, each mask is of shape (D, H, W)
    """
    min_index = np.inf
    max_index = -np.inf
    for volume_mask in volume_mask_list:
        z_any = [np.any(m) for m in volume_mask]
        z_range = np.where(z_any)[0]

        if min_index > min(z_range):
            min_index = min(z_range)
        if max_index < max(z_range):
            max_index = max(z_range)

    img = volume_img[min_index:max_index]
    mask = []
    for volume_mask in volume_mask_list:
        mask.append(volume_mask[min_index:max_index])

    # combine all rois into one roi
    mask = (np.array(mask).sum(axis=0)) > 0

    def fz(k):
        plt.figure(figsize=(15, 15))
        plt.imshow(img[k], cmap='gray', alpha=0.7)
        # combine all rois into one roi
        plt.imshow(mask[k], alpha=0.3)

    w.interact(fz, k=w.IntSlider(min=0, max=img.shape[0] - 1, step=1, value=0))



