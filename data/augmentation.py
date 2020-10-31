import numpy as np
import time
from scipy.ndimage import rotate, gaussian_filter, map_coordinates
from skimage.transform import resize
from skimage import exposure
import cv2
import torch
import importlib
from torchvision.transforms import Compose

GLOBAL_RANDOM_STATE = np.random.RandomState(0)  # int(time.time())


########################################################################################################################
#                                  No random value, apply to input image(not mask)                                     #
########################################################################################################################
class MinMaxScaling(object):
    def __init__(self, minval, maxval, use_bodymask=False, **kwargs):
        super(MinMaxScaling, self).__init__()
        self.minval = minval
        self.maxval = maxval
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            Scale the image in range [minval, maxval].
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        print(f"\tMinMax - [min: {self.minval}, max: {self.maxval}]")
        assert 'img' in volume and volume[
            'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on MinMaxScaling"

        volume_img = volume['img']

        if not self.use_bodymask:
            volume_img = (volume_img - np.min(volume_img)) * (self.maxval - self.minval) / (
                    np.max(volume_img) - np.min(volume_img)) + self.minval
            volume['img'] = volume_img
        else:
            assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                          "'bodymask' and it is 3d, " \
                                                                          "exit on MinMaxScaling "
            volume_body_mask = volume['bodymask']
            volume_body_img = volume_img[volume_body_mask > 0]
            volume_body_img = (volume_body_img - np.min(volume_body_img)) * (self.maxval - self.minval) / (
                    np.max(volume_body_img) - np.min(volume_body_img)) + self.minval
            volume['img'][volume_body_mask > 0] = volume_body_img
            volume['img'][volume_body_mask == 0] = volume_body_img.min()
        return volume


class ClipValue(object):
    def __init__(self, minval, maxval, use_bodymask=False, **kwargs):
        super(ClipValue, self).__init__()
        self.minval = minval
        self.maxval = maxval
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            clip the volume to range [minval, maxval]
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        print(f"\tClip - [min: {self.minval}, max: {self.maxval}]")
        assert 'img' in volume and volume[
            'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on ClipValue"
        volume_img = volume['img']

        if not self.use_bodymask:
            volume_img = np.clip(volume_img, a_min=self.minval, a_max=self.maxval)
            volume['img'] = volume_img
        else:
            assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                          "'bodymask' and it is 3d, exit on ClipValue "
            volume_body_mask = volume['bodymask']
            volume_body_img = volume_img[volume_body_mask > 0]
            volume_body_img = np.clip(volume_body_img, a_min=self.minval, a_max=self.maxval)
            volume['img'][volume_body_mask > 0] = volume_body_img
            volume['img'][volume_body_mask == 0] = volume_body_img.min()

        return volume


class Normalize(object):
    def __init__(self, mean=0, std=1, use_bodymask=False, **kwargs):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            make volume applied to standard normal distribution.
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        print(f"\tNormalize")
        assert 'img' in volume and volume[
            'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on Normalize"
        volume_img = volume['img']

        if not self.use_bodymask:
            volume_img = (volume_img - np.mean(volume_img)) / (
                        np.std(volume_img) + np.finfo(float).eps)
            volume_img = volume_img * self.std + self.mean
            volume['img'] = volume_img
        else:
            assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                          "'bodymask' and it is 3d, exit on Normalize "
            volume_body_mask = volume['bodymask']
            volume_body_img = volume_img[volume_body_mask > 0]
            volume_body_img = (volume_body_img - np.mean(volume_body_img)) / (np.std(volume_body_img) + np.finfo(float).eps)
            volume_body_img = volume_body_img * self.std + self.mean
            volume['img'][volume_body_mask > 0] = volume_body_img
            volume['img'][volume_body_mask == 0] = volume_body_img.min()
        return volume


########################################################################################################################
#                                  With random value, apply to input image(not mask)                                   #
########################################################################################################################
class RandomContrast(object):
    def __init__(self, alpha_range=(0.5, 1.5), exe_prob=0.5, use_bodymask=False, **kwargs):
        super(RandomContrast, self).__init__()
        self.alpha_range = alpha_range
        self.exe_prob = exe_prob
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            Change volume contrast randomly.
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume[
                'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on RandomContrast"
            volume_img = volume['img']
            alpha = GLOBAL_RANDOM_STATE.uniform(*self.alpha_range)
            print(f"\tRandomContrast - [alpha: {alpha}]")

            if not self.use_bodymask:
                mean_value = np.mean(volume_img)
                volume_img = (mean_value + (volume_img - mean_value) * alpha)
                volume['img'] = volume_img
            else:
                assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                              "'bodymask' and it is 3d, " \
                                                                              "exit on RandomContrast "
                volume_body_mask = volume['bodymask']
                volume_body_img = volume_img[volume_body_mask > 0]
                mean_value = np.mean(volume_body_img)
                volume_body_img = (mean_value + (volume_body_img - mean_value) * alpha)
                volume['img'][volume_body_mask > 0] = volume_body_img
                volume['img'][volume_body_mask == 0] = volume_body_img.min()
        return volume


class AdditiveGaussianNoise(object):
    def __init__(self, mean=0, std_range=(1, 5), exe_prob=0.5, use_bodymask=False, **kwargs):
        super(AdditiveGaussianNoise, self).__init__()
        self.mean = mean
        self.std_range = std_range
        self.exe_prob = exe_prob
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            add gaussian noise to input volume
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume[
                'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on AdditiveGaussianNoise"
            volume_img = volume['img']
            std = GLOBAL_RANDOM_STATE.uniform(*self.std_range)
            print(f"\tAdditiveGaussianNoise - [mean: {self.mean}, std: {std}]")

            if not self.use_bodymask:
                gaussian_noise = GLOBAL_RANDOM_STATE.normal(self.mean, std, size=volume_img.shape)
                volume_img = volume_img + gaussian_noise
                volume['img'] = volume_img
            else:
                assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                              "'bodymask' and it is 3d, " \
                                                                              "exit on AdditiveGaussianNoise "
                volume_body_mask = volume['bodymask']
                volume_body_img = volume_img[volume_body_mask > 0]
                gaussian_noise = GLOBAL_RANDOM_STATE.normal(self.mean, std, size=volume_body_img.shape)
                volume_body_img = volume_body_img + gaussian_noise
                volume['img'][volume_body_mask > 0] = volume_body_img
                volume['img'][volume_body_mask == 0] = volume_body_img.min()
        return volume


class AdditivePoissonNoise(object):
    def __init__(self, lbd_range=(0, 1.0), exe_prob=0.5, use_bodymask=False, **kwargs):
        super(AdditivePoissonNoise, self).__init__()
        self.lbd_range = lbd_range
        self.exe_prob = exe_prob
        self.use_bodymask = use_bodymask

    def __call__(self, volume):
        """
            add poission noise to input volume
            If self.use_bodymask is True, then, only the voxel extracted by the mask are involved in the calculation,
            remaining area are set to the min value.
        params:
            volume: dict:
                key:
                    img: [D, H, W] -> numpy ndarray
                    body_mask[optional]: [D, H, W] -> numpy ndarray
                    ...
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on AdditivePoissonNoise "
            volume_img = volume['img']
            lbd = GLOBAL_RANDOM_STATE.uniform(*self.lbd_range)
            print(f"\tAdditivePoissonNoise - [lambda: {lbd}]")

            if not self.use_bodymask:
                poisson_noise = GLOBAL_RANDOM_STATE.poisson(lbd, size=volume_img.shape)
                volume_img = volume_img + poisson_noise
                volume['img'] = volume_img
            else:
                assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                              "'bodymask' and it is 3d, " \
                                                                              "exit on AdditivePoissonNoise "
                volume_body_mask = volume['bodymask']
                volume_body_img = volume_img[volume_body_mask > 0]
                poisson_noise = GLOBAL_RANDOM_STATE.poisson(lbd, size=volume_body_img.shape)
                volume_body_img = volume_body_img + poisson_noise
                volume['img'][volume_body_mask > 0] = volume_body_img
                volume['img'][volume_body_mask == 0] = volume_body_img.min()
        return volume


########################################################################################################################
#                                  Histogram, apply to input image, with mask or not                                   #
#   Adaptive Histogram Equalization (AHE) and Contrast Limited Adaptive Histogram Equalization (CLAHE) are not proper  #
########################################################################################################################
class ShiftHistMode(object):
    def __init__(self, mode_hu_range, bins='auto', exe_prob=0.5, target_roi_idx_list=None, use_mask=False, **kwargs):
        super(ShiftHistMode, self).__init__()
        self.mode_hu_range = mode_hu_range
        self.bins = bins
        self.exe_prob = exe_prob
        self.target_roi_idx_list = target_roi_idx_list
        self.use_mask = use_mask

    def __call__(self, volume):
        """
            shift original hu histogram mode to desired range. If mask is not None, process entire volume_img
            else, only process the area masked out by volume mask.
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on ShiftHistMode "
            volume_img = volume['img']
            target_mode = GLOBAL_RANDOM_STATE.uniform(*self.mode_hu_range)
            print(f"\tShiftHistMode: [mode: {target_mode}, roi_list: {self.target_roi_idx_list}]")

            if not self.use_mask:
                mode_value = self.get_mode_value(volume_img)
                volume_img = volume_img - mode_value + target_mode
                volume['img'] = volume_img
            else:
                assert 'mask' in volume and volume['mask'].ndim == 4, "Input volume must have attribute 'mask' and it " \
                                                                      "is 4d, exit on ShiftHistMode "
                volume_mask = volume['mask']
                # print('target_mode: ', target_mode)
                for roi_idx in self.target_roi_idx_list:
                    masked_area = volume_img[volume_mask[roi_idx] > 0]  # bool array could be used to extract masked area
                    mode_value = self.get_mode_value(masked_area)
                    volume_img[volume_mask[roi_idx] > 0] = masked_area - mode_value + target_mode
                volume['img'] = volume_img
        return volume

    def get_mode_value(self, values):
        hist, bin_edges = np.histogram(values, bins=self.bins)
        mode_bin_idx = np.argmax(hist)
        mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2
        return mode_value


class ContrastStretchingOnMode(object):
    def __init__(self, alpha_range, bins='auto', exe_prob=0.5, target_roi_idx_list=None, use_mask=False, **kwargs):
        super(ContrastStretchingOnMode, self).__init__()
        self.alpha_range = alpha_range
        self.bins = bins
        self.exe_prob = exe_prob
        self.target_roi_idx_list = target_roi_idx_list
        self.use_mask = use_mask

    def __call__(self, volume):
        """
            change contrast of img based on mode value.
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on ContrastStretchingOnMode "
            volume_img = volume['img']
            alpha = GLOBAL_RANDOM_STATE.uniform(*self.alpha_range)
            print(f"\tContrastStretchingOnMode: [alpha: {alpha}, roi_list: {self.target_roi_idx_list}]")

            if not self.use_mask:
                mode_value = self.get_mode_value(volume_img)
                volume_img = (volume_img - mode_value) * alpha + mode_value
                volume['img'] = volume_img
            else:
                assert 'mask' in volume and volume['mask'].ndim == 4, "Input volume must have attribute 'mask' and it " \
                                                                      "is 4d, exit on ContrastStretchingOnMode "
                volume_mask = volume['mask']
                for roi_idx in self.target_roi_idx_list:
                    masked_area = volume_img[volume_mask[roi_idx] > 0]  # bool array could be used to extract masked area
                    mode_value = self.get_mode_value(masked_area)
                    volume_img[volume_mask[roi_idx] > 0] = (masked_area - mode_value) * alpha + mode_value
                volume['img'] = volume_img
        return volume

    def get_mode_value(self, values):
        hist, bin_edges = np.histogram(values, bins=self.bins)
        mode_bin_idx = np.argmax(hist)
        mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2
        return mode_value


class ContrastStretching(object):
    def __init__(self, out_range, outlier_percentile=5, exe_prob=0.5, target_roi_idx_list=None, use_mask=False, **kwargs):
        super(ContrastStretching, self).__init__()
        self.out_range = out_range
        self.outlier_percentile = outlier_percentile
        self.exe_prob = exe_prob
        self.target_roi_idx_list = target_roi_idx_list
        self.use_mask = use_mask

    def __call__(self, volume):
        """
            http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on ContrastStretching "
            volume_img = volume['img']
            print(f"\tContrastStretching: [roi_list: {self.target_roi_idx_list}]")

            if not self.use_mask:
                p_1, p_2 = self.get_stretch_range(volume_img)
                volume_img = exposure.rescale_intensity(volume_img, in_range=(p_1, p_2), out_range=self.out_range)
                volume['img'] = volume_img
            else:
                assert 'mask' in volume and volume['mask'].ndim == 4, "Input volume must have attribute 'mask' and it " \
                                                                      "is 4d, exit on ContrastStretching "
                volume_mask = volume['mask']
                for roi_idx in self.target_roi_idx_list:
                    masked_area = volume_img[volume_mask[roi_idx] > 0]  # bool array could be used to extract masked area
                    p_1, p_2 = self.get_stretch_range(masked_area)
                    volume_img[volume_mask[roi_idx] > 0] = exposure.rescale_intensity(masked_area, in_range=(p_1, p_2),
                                                                                      out_range=self.out_range)
                volume['img'] = volume_img
        return volume

    def get_stretch_range(self, values):
        p_1 = np.percentile(values, self.outlier_percentile)
        p_2 = np.percentile(values, 100 - self.outlier_percentile)
        return p_1, p_2


class HistogramEqualization(object):
    def __init__(self, out_range, exe_prob=0.5, target_roi_idx_list=None, use_mask=False, **kwargs):
        super(HistogramEqualization, self).__init__()
        self.out_range = out_range
        self.exe_prob = exe_prob
        self.target_roi_idx_list = target_roi_idx_list
        self.use_mask = use_mask

    def __call__(self, volume):
        """
            histogram equalization: http://homepages.inf.ed.ac.uk/rbf/HIPR2/histeq.htm
            if out_range is None: output_range is set to original volume_img range.
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on HistogramEqualization "
            volume_img = volume['img']
            print(f"\tHistogramEqualization: [roi_list: {self.target_roi_idx_list}]")

            if not self.use_mask:
                volume_img = exposure.equalize_hist(volume_img)
                volume_img = exposure.rescale_intensity(volume_img, in_range=(0, 1), out_range=self.out_range)
                volume['img'] = volume_img
            else:
                assert 'mask' in volume and volume['mask'].ndim == 4, "Input volume must have attribute 'mask' and it " \
                                                                      "is 4d, exit on HistogramEqualization "
                volume_mask = volume['mask']
                # print("histogram_equalization")
                for roi_idx in self.target_roi_idx_list:
                    masked_area = volume_img[volume_mask[roi_idx] > 0]  # bool array could be used to extract masked area
                    masked_area = exposure.equalize_hist(masked_area)
                    volume_img[volume_mask[roi_idx] > 0] = exposure.rescale_intensity(masked_area, in_range=(0, 1),
                                                                                      out_range=self.out_range)
                volume['img'] = volume_img
        return volume


########################################################################################################################
#                                  With random value, apply to input image and mask                                    #
########################################################################################################################
class RandomRotate(object):
    def __init__(self, angle_range=(-5, 5), axes=None, exe_prob=0.5, **kwargs):
        super(RandomRotate, self).__init__()
        self.angle_range = angle_range
        self.axes = axes
        self.exe_prob = exe_prob

    def __call__(self, volume):
        """
            rotate volume randomly
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume[
                'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on RandomRotate"
            assert 'mask' in volume and volume[
                'mask'].ndim == 4, "Input volume must have attribute 'mask' and it is 4d, exit on RandomRotate"

            volume_img = volume['img']
            volume_mask = volume['mask']
            if self.axes == 'yx':
                axes = (1, 2)
            elif self.axes == 'zy':
                axes = (0, 1)
            elif self.axes == 'zx':
                axes = (0, 2)
            else:
                axes_list = [(0, 1), (0, 2), (1, 2)]
                axes = axes_list[GLOBAL_RANDOM_STATE.randint(0, len(axes_list))]
            # print(f"RandomRotate plane: {axes}")
            angle = GLOBAL_RANDOM_STATE.uniform(*self.angle_range)
            print(f"\tRandomRotate - [axis_plane: {axes}, angle: {angle}]")

            min_img = np.min(volume_img)
            volume_img = rotate(volume_img, angle=angle, axes=axes, reshape=False, mode='constant', cval=min_img)

            volume_mask = np.stack(
                [rotate(mask_one_roi, angle=angle, axes=axes, reshape=False, mode='constant', cval=0)
                 for mask_one_roi in volume_mask],
                axis=0)
            # make mask binary
            volume_mask = (volume_mask > 0.5).astype(np.uint8)

            volume['img'] = volume_img
            volume['mask'] = volume_mask
        return volume


class RandomAffine(object):
    def __init__(self, delta_range=(-15, 15), axes=None, exe_prob=0.5, **kwargs):
        super(RandomAffine, self).__init__()
        self.delta_range = delta_range
        self.axes = axes
        self.exe_prob = exe_prob

    def __call__(self, volume):
        """
            random affine volume
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume[
                'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on RandomAffine"
            assert 'mask' in volume and volume[
                'mask'].ndim == 4, "Input volume must have attribute 'mask' and it is 4d, exit on RandomAffine"

            volume_img = volume['img']
            volume_mask = volume['mask']
            if self.axes == 'yx':
                swap_axes = None
            elif self.axes == 'zy':
                swap_axes = (0, 2)
            elif self.axes == 'zx':
                swap_axes = (0, 1)
            else:
                swap_axes_list = [None, (0, 2), (0, 1)] # None -> yx, (0, 2) -> zy, (0, 1) -> zx
                idx = GLOBAL_RANDOM_STATE.randint(0, len(swap_axes_list))
                swap_axes = swap_axes_list[idx]

            print(f"\tRandomAffine - [swap_axes: {swap_axes}]")

            # opencv uses xy order
            shape_2D_xy = np.asarray(self.swap_volume_axes(volume_img, swap_axes).shape[-2:][::-1])
            # get affine transformation matrix
            M = self.get_affine_matrix_2D(shape_2D_xy, self.delta_range)

            # apply affine transformation to volume_img
            volume_img = volume_img.astype(np.float)
            volume_img = self.apply_affine_transformation(volume_img, swap_axes, M, tuple(shape_2D_xy),
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=np.min(volume_img))

            volume_mask = np.stack([self.apply_affine_transformation(mask_one_roi, swap_axes, M, tuple(shape_2D_xy),
                                                                     borderMode=cv2.BORDER_CONSTANT,
                                                                     borderValue=0)
                                    for mask_one_roi in volume_mask],
                                   axis=0)
            # make mask binary
            volume_mask = (volume_mask > 0.5).astype(np.uint8)

            volume['img'] = volume_img
            volume['mask'] = volume_mask
        return volume

    def swap_volume_axes(self, volume, swap_axes):
        if swap_axes is not None:
            volume = np.swapaxes(volume, *swap_axes)
        return volume

    def get_affine_matrix_2D(self, shape_2D, delta_range):
        center_pt = shape_2D // 2
        src = np.asarray([center_pt + center_pt // 2,
                          [center_pt[0] + center_pt[0] // 2, center_pt[1] - center_pt[1] // 2],
                          center_pt - center_pt // 2],
                         dtype=np.float32)
        delta = GLOBAL_RANDOM_STATE.uniform(*delta_range, size=src.shape).astype(np.float32)
        dst = (src + delta).astype(np.float32)
        M = cv2.getAffineTransform(src, dst)
        return M

    def apply_affine_transformation(self, volume, swap_axes, M, size, borderMode, borderValue):
        # swap axes to make affine plane
        volume = self.swap_volume_axes(volume, swap_axes)
        # do affine transformation operation
        volume = np.stack(
            [cv2.warpAffine(slice_2D, M, size, borderMode=borderMode, borderValue=borderValue)
             if np.any(slice_2D) else slice_2D for slice_2D in volume], axis=0)
        # swap back to original axes order
        volume = self.swap_volume_axes(volume, swap_axes)
        return volume


class ElasticDeformation(object):
    def __init__(self, grid_size=4, sigma=3, alpha=15, exe_prob=0.5, **kwargs):
        self.grid_size = grid_size
        self.sigma = sigma
        self.alpha = alpha
        self.exe_prob = exe_prob

    def __call__(self, volume):
        """
                Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
                Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        if GLOBAL_RANDOM_STATE.uniform() < self.exe_prob:
            assert 'img' in volume and volume[
                'img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, exit on ElasticDeformation"
            assert 'mask' in volume and volume[
                'mask'].ndim == 4, "Input volume must have attribute 'mask' and it is 4d, exit on ElasticDeformation"

            volume_img = volume['img']
            volume_mask = volume['mask']

            print(f"\tElasticDeformation")

            volume_shape = np.array(volume_img.shape)
            # diff
            dz, dy, dx = [gaussian_filter(GLOBAL_RANDOM_STATE.randn(*(volume_shape // self.grid_size)), self.sigma) * self.alpha for _
                          in
                          range(3)]
            dz, dy, dx = [resize(d, volume_shape) for d in [dz, dy, dx]]
            dz = dz * 0.5
            z_dim, y_dim, x_dim = volume_shape
            # base coordinate
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = [z + dz, y + dy, x + dx]

            # img
            min_img = np.min(volume_img)
            # map coordinate
            volume_img = map_coordinates(volume_img, indices, order=0, mode='constant', cval=min_img)

            volume_mask = np.stack([map_coordinates(mask_one_roi, indices, order=0, mode='constant', cval=0)
                                    for mask_one_roi in volume_mask],
                                   axis=0)
            # make mask binary
            volume_mask = (volume_mask > 0.5).astype(np.uint8)

            volume['img'] = volume_img
            volume['mask'] = volume_mask
        return volume


class MaskArrayToMask(object):
    def __init__(self, **kwargs):
        super(MaskArrayToMask, self).__init__()

    def __call__(self, volume):
        """
            combine multiple binary masks to one mask
        """
        print(f"\tMaskArrayToMask")
        assert 'mask' in volume and volume[
            'mask'].ndim == 4, "Input volume must have attribute 'mask' and it is 4d, exit on ToTensor"
        volume_mask = volume['mask']
        num_mask, D, H, W = volume_mask.shape

        mask = np.zeros((D, H, W))
        for i in range(num_mask):
            # background is label 0
            mask[volume_mask[i] > 0] = i + 1

        volume['mask'] = mask
        return volume


class ToTensor(object):
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, **kwargs):
        super(ToTensor, self).__init__()

    def __call__(self, volume):
        """
            convert value of volume from numpy ndarray to torch tensor
        prams:
            volume: dict
                key:
                    img: [D, H, W] -> numpy ndarray
                    mask: [N, D, H, W], N is num of rois -> numpy ndarray
        """
        print(f"\tToTensor")
        if 'img' in volume:
            assert 'img' in volume and volume['img'].ndim == 3, "Input volume must have attribute 'img' and it is 3d, " \
                                                                "exit on ToTensor "
            volume['img'] = torch.from_numpy(volume['img'].astype(dtype=np.float32))

        if 'bodymask' in volume:
            assert 'bodymask' in volume and volume['bodymask'].ndim == 3, "Input volume must have attribute " \
                                                                          "'bodymask' and it is 4d, exit on ToTensor "
            volume['bodymask'] = torch.from_numpy(volume['bodymask'].astype(dtype=np.float32))

        if 'mask' in volume:
            assert 'mask' in volume and volume['mask'].ndim == 4, "Input volume must have attribute 'mask' and it is " \
                                                                  "4d, exit on ToTensor "
            volume['mask'] = torch.from_numpy(volume['mask'].astype(dtype=np.uint8))

        if 'mask_flag' in volume:
            volume['mask_flag'] = torch.tensor(volume['mask_flag'])

        return volume


class Transformer:
    def __init__(self, augment_config):
        self.augment_config = augment_config
        self.module = importlib.import_module('data.augmentation')

    def play_transform(self):
        return self._create_transform()

    def _create_transform(self):
        return Compose([
            self._create_augmentation(config) for i, config in enumerate(self.augment_config)])

    def _create_augmentation(self, config):
        # print(config['name'])
        aug_class = getattr(self.module, config['name'])
        return aug_class(**config)
