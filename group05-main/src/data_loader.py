"""MONAI transforms for the BraTS lab."""

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    ToTensord,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
)


CLASS_NAMES = ("edema", "non_enhancing_tumor", "enhancing_tumor")
DATA_CONFIG = {
    "label_classes": "0=background,1=edema,2=non_enhancing_tumor,3=enhancing_tumor",
    "data_normalization": "monai_scale_intensity_per_channel",
    "data_sampling": "monai_rand_crop_pos_neg_label",
    "data_augmentations": "flips_xyz, rotation, scaling, gaussian_noise, gamma_contrast",
}


def basic_train_transforms(patch_size, patches_per_case):
    """Load images and masks, then ask MONAI to sample training patches."""
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image", channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=patches_per_case,
        ),
    ]
    steps.append(ToTensord(keys="image", dtype=torch.float32))
    steps.append(ToTensord(keys="label", dtype=torch.int64))
    return Compose(steps)


def advanced_train_transforms(patch_size, patches_per_case):
    """Copy the basic loader and add augmentations at YOUR CODE HERE."""
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image", channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=patches_per_case,
        ),
        # --- spatial (image + label) ---
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.26, 0.26, 0.26),      # about ±15° per axis
            scale_range=(0.1, 0.1, 0.1),          # ±10% scaling
            mode=("bilinear", "nearest"),         # nearest keeps labels at 0,1,2,3
            padding_mode="border",
        ),

        # --- intensity (image only) ---
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),

        # STEP 1 — YOUR CODE HERE. The paper changes the shape and position
        # of training patches with rotation, scaling, and mirroring. Apply
        # each spatial change to the MRI and its mask together so they stay
        # aligned (p. 17 and Appendix D).
        # It also varies MRI appearance with noise, blur, brightness,
        # contrast, and gamma. Apply intensity changes only to the image;
        # the mask must keep its class numbers.
    ]

    steps.append(ToTensord(keys="image", dtype=torch.float32))
    steps.append(ToTensord(keys="label", dtype=torch.int64))
    return Compose(steps)


def val_transforms():
    """Load the full image for sliding-window validation; do not crop."""
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image", channel_wise=True),
    ]
    steps.append(ToTensord(keys="image", dtype=torch.float32))
    steps.append(ToTensord(keys="label", dtype=torch.int64))
    return Compose(steps)
