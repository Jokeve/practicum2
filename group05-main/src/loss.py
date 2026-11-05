"""Baseline loss and a separate nnU-Net deep-supervision template."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from monai.losses import DiceCELoss



LOSS_CONFIG = {"loss": "cross_entropy"}
NNUNET_LOSS_CONFIG = {"loss": "dice_ce_with_deep_supervision"}


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare four class logits with the raw one-channel label map."""
    return F.cross_entropy(logits, target[:, 0].long())


def nnunet_loss(outputs: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Step 3: implement the paper's Dice + CE loss and deep supervision."""
    # YOUR CODE HERE. Read pp. 16–17 and 25 of the paper. Compare the
    # full-resolution prediction with the original label, then compare the
    # half-resolution prediction with a nearest-neighbor resized label.
    # Give the full-resolution loss twice the weight of the other loss.

    full_logits, half_logits = outputs
    
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
    half_target = F.interpolate(target.float(), size=half_logits.shape[2:], mode="nearest").long()

    full_loss = criterion(full_logits, target)
    half_loss = criterion(half_logits, half_target)

    return (full_loss + 0.5 * half_loss) / 1.5
