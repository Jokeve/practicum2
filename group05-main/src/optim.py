"""Baseline optimizer and a separate nnU-Net scheduler template."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


OPTIM_CONFIG = {"optimizer": "adam", "schedule": "none"}
NNUNET_OPTIM_CONFIG = {"optimizer": "sgd_nesterov (exercise)", "schedule": "polynomial (exercise)"}


def build_optimizer(
    parameters: Iterable[nn.Parameter], learning_rate: float, epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Return an optimizer and optional per-epoch scheduler."""
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    return optimizer, None


def build_nnunet_optimizer(
    parameters: Iterable[nn.Parameter], learning_rate: float, epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Step 4: implement the paper's optimizer and learning-rate rule."""
    # YOUR CODE HERE. Read pp. 16–19 and 24–26 of the paper. Create SGD
    # with Nesterov momentum, then a polynomial learning-rate scheduler.
    # Return both; main.py advances the scheduler after each epoch.
    raise NotImplementedError("Implement build_nnunet_optimizer in step 4")
