"""Per-case validation metrics and W&B image overlays."""

from __future__ import annotations

import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch import nn
from torch.utils.data import DataLoader

from src.data_loader import CLASS_NAMES, val_transforms


def dice_per_class(logits: torch.Tensor, target: torch.Tensor) -> list[float]:
    prediction = logits.argmax(dim=1)
    truth = target[:, 0]
    values = []
    for class_id in (1, 2, 3):
        pred = prediction == class_id
        actual = truth == class_id
        denominator = int(pred.sum().item() + actual.sum().item())
        values.append(1.0 if denominator == 0 else 2 * (pred & actual).sum().item() / denominator)
    return values


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, patch_size: tuple[int, int, int],
    sw_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = sliding_window_inference(image, patch_size, sw_batch_size, model, overlap=0.25)
            scores = dice_per_class(logits, label)
            rows.append({"case_id": batch["case_id"][0], **dict(zip(CLASS_NAMES, scores)), "mean": sum(scores) / 3})
    means = {name: float(np.mean([row[name] for row in rows])) for name in (*CLASS_NAMES, "mean")}
    return means, rows


def overlay_for_case(
    case: dict, model: nn.Module, device: torch.device, args: argparse.Namespace
):
    """Create one annotated MRI slice for a W&B Table."""
    import wandb

    sample = val_transforms()(case)
    image, label = sample["image"].numpy(), sample["label"].numpy()
    tensor = torch.from_numpy(image[None]).to(device)
    model.eval()
    with torch.inference_mode():
        logits = sliding_window_inference(tensor, args.patch_size, args.sw_batch_size, model, overlap=0.25)
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    slice_idx = int(np.argmax((label[0] > 0).sum(axis=(0, 1))))
    display_channel = 2 if isinstance(case["image"], str) else 1
    display = image[display_channel, :, :, slice_idx]
    display = np.clip((display - display.min()) / max(float(np.ptp(display)), 1e-6), 0, 1)

    class_labels = {0: "background", 1: "edema", 2: "non-enhancing tumor", 3: "enhancing tumor"}
    return wandb.Image(
        display,
        caption=f"{case['case_id']}: MRI channel {display_channel}; slice {slice_idx}",
        masks={
            "ground_truth": {"mask_data": label[0, :, :, slice_idx].astype(np.uint8), "class_labels": class_labels},
            "prediction": {"mask_data": prediction[:, :, slice_idx], "class_labels": class_labels},
        },
    )
