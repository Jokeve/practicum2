"""BraTS lab training and validation pipeline.

Data loading, network architecture, and loss live in src/ modules. Students
change those modules and import their updated functions here.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader

from src.data_loader import (
    DATA_CONFIG, CLASS_NAMES, advanced_train_transforms, val_transforms,
)
from src.evaluation import evaluate, overlay_for_case
from src.loss import LOSS_CONFIG, NNUNET_LOSS_CONFIG, nnunet_loss, segmentation_loss
from src.model import MODEL_CONFIG, NNUNET_MODEL_CONFIG, BasicUNet, NNUNet
from src.optim import OPTIM_CONFIG, NNUNET_OPTIM_CONFIG, build_nnunet_optimizer, build_optimizer


VALIDATION_FRACTION = 0.2
VALIDATION_SPLIT_SEED = 42


# Training loop, logging, and command-line entry point remain in main.py.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    worker_seed_value = torch.initial_seed() % (2**32)
    random.seed(worker_seed_value)
    np.random.seed(worker_seed_value)


def find_nifti(folder: Path, name: str) -> Path:
    for extension in (".nii.gz", ".nii"):
        path = folder / f"{name}{extension}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name}.nii[.gz] in {folder}")


def list_cases(data_root: Path, modalities: tuple[str, ...], label_suffix: str) -> list[dict]:
    """Make the image/label dictionaries passed directly to MONAI CacheDataset."""
    images_dir = data_root / "imagesTr"
    if images_dir.is_dir():
        cases = []
        for image in sorted(images_dir.glob("*.nii*")):
            case_id = image.name.removesuffix(".nii.gz").removesuffix(".nii")
            cases.append({
                "image": str(image),
                "label": str(find_nifti(data_root / "labelsTr", case_id)),
                "case_id": case_id,
            })
    else:
        cases = []
        for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
            cases.append({
                "image": [str(find_nifti(folder, f"{folder.name}_{suffix}")) for suffix in modalities],
                "label": str(find_nifti(folder, f"{folder.name}_{label_suffix}")),
                "case_id": folder.name,
            })
    if not cases:
        raise FileNotFoundError(f"No patient images found in {data_root}")
    return cases


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Use the same patient-level validation cases in every run."""
    shuffled = sorted(cases, key=lambda case: case["case_id"])
    random.Random(VALIDATION_SPLIT_SEED).shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * VALIDATION_FRACTION)))
    return shuffled[n_val:], shuffled[:n_val]


def git_revision() -> tuple[str | None, bool | None]:
    """Record the checked-out revision and whether the working tree was edited."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_training(args: argparse.Namespace) -> None:
    import wandb

    seed_everything(args.seed)
    cases = list_cases(args.data_root, args.modality_suffixes, args.label_suffix)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    train_cases, val_cases = split_cases(cases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = CacheDataset(
        data=train_cases,
        transform=advanced_train_transforms(args.patch_size, args.patches_per_case),
        cache_num=1, num_workers=0, progress=False,
    )
    val_set = CacheDataset(
        data=val_cases, transform=val_transforms(),
        cache_num=1, num_workers=0, progress=False,
    )
    loader_kwargs = {"num_workers": args.num_workers, "worker_init_fn": worker_seed}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_kwargs)
    model = (NNUNet(args.patch_size, args.base_channels) if args.model == "nnunet"
             else BasicUNet(args.base_channels)).to(device)
    loss_fn = nnunet_loss if args.loss == "nnunet" else segmentation_loss
    optimizer_fn = build_nnunet_optimizer if args.optimizer == "nnunet" else build_optimizer
    optimizer, scheduler = optimizer_fn(model.parameters(), args.learning_rate, args.epochs)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["modality_suffixes"] = list(args.modality_suffixes)
    config["patch_size"] = list(args.patch_size)
    if isinstance(cases[0]["image"], str):
        config["dataset_layout"] = "four_channel_nifti"
        metadata_path = args.data_root / "dataset.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        channel_names = metadata.get("modality", metadata.get("channel_names", {}))
        config["modality_order"] = [channel_names.get(str(i), f"channel_{i}") for i in range(4)]
    else:
        config["dataset_layout"] = "one_folder_per_case"
        config["modality_order"] = list(args.modality_suffixes)
    commit, dirty = git_revision()
    model_config = NNUNET_MODEL_CONFIG if args.model == "nnunet" else MODEL_CONFIG
    loss_config = NNUNET_LOSS_CONFIG if args.loss == "nnunet" else LOSS_CONFIG
    optim_config = NNUNET_OPTIM_CONFIG if args.optimizer == "nnunet" else OPTIM_CONFIG
    config.update({
        **DATA_CONFIG, **model_config, **loss_config, **optim_config, "device": str(device),
        "model_levels": model.depth + 1 if args.model == "nnunet" else 3,
        "effective_batch_size": args.batch_size * args.patches_per_case,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "git_commit": commit, "git_dirty": dirty,
        "train_cases": len(train_cases), "val_cases": len(val_cases),
        "validation_fraction": VALIDATION_FRACTION,
        "validation_split_seed": VALIDATION_SPLIT_SEED,
    })
    with wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
        name=args.run_name, mode=args.wandb_mode, config=config,
    ) as run:
        output_dir = args.output_dir / run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        split_path = output_dir / "split.json"
        split_path.write_text(json.dumps({"train": [c["case_id"] for c in train_cases], "val": [c["case_id"] for c in val_cases]}, indent=2))
        split_artifact = wandb.Artifact(f"split-{run.id}", type="split")
        split_artifact.add_file(str(split_path))
        run.log_artifact(split_artifact)

        best_score = -1.0
        best_epoch = 0
        peak_gpu_memory_gb = 0.0
        checkpoint_path = output_dir / "best.pt"
        started = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            losses = []
            for batch in train_loader:
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(image, return_aux=True) if args.loss == "nnunet" else model(image)
                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            metrics = {
                "epoch": epoch,
                "train/loss": float(np.mean(losses)),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "time/elapsed_seconds": time.monotonic() - started,
            }
            if epoch % args.val_interval == 0 or epoch == args.epochs:
                scores, _ = evaluate(model, val_loader, device, args.patch_size, args.sw_batch_size)
                metrics.update({f"val/dice_{name}": value for name, value in scores.items()})
                if scores["mean"] > best_score:
                    best_score = scores["mean"]
                    best_epoch = epoch
                    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, checkpoint_path)
            metrics["time/elapsed_seconds"] = time.monotonic() - started
            if device.type == "cuda":
                current_peak = torch.cuda.max_memory_allocated(device) / 1e9
                metrics["resources/peak_gpu_memory_gb"] = current_peak
                peak_gpu_memory_gb = max(peak_gpu_memory_gb, current_peak)
            run.log(metrics)
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch}/{args.epochs}: train loss {metrics['train/loss']:.4f}" + (f", val Dice {scores['mean']:.4f}" if "val/dice_mean" in metrics else ""))

        final_scores = scores
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_scores, rows = evaluate(model, val_loader, device, args.patch_size, args.sw_batch_size)
        selected = {min(rows, key=lambda row: row["mean"])["case_id"], max(rows, key=lambda row: row["mean"])["case_id"]}
        lookup = {case["case_id"]: case for case in val_cases}
        table = wandb.Table(columns=["case_id", *CLASS_NAMES, "mean", "overlay"])
        for row in rows:
            overlay = overlay_for_case(lookup[row["case_id"]], model, device, args) if row["case_id"] in selected else None
            table.add_data(row["case_id"], *(row[name] for name in CLASS_NAMES), row["mean"], overlay)
        run.log({"validation/cases": table})
        checkpoint_artifact = wandb.Artifact(f"best-model-{run.id}", type="model")
        checkpoint_artifact.add_file(str(checkpoint_path))
        run.log_artifact(checkpoint_artifact)
        summary = {
            "best_epoch": best_epoch,
            "best_mean_dice": best_score,
            "final_epoch": args.epochs,
            "final_mean_dice": final_scores["mean"],
            "total_seconds": time.monotonic() - started,
            **{f"best_dice_{name}": best_scores[name] for name in CLASS_NAMES},
        }
        if device.type == "cuda":
            summary["peak_gpu_memory_gb"] = max(peak_gpu_memory_gb, torch.cuda.max_memory_allocated(device) / 1e9)
        run.summary.update(summary)
        print(f"Final epoch {args.epochs}: mean tumor-class Dice {final_scores['mean']:.4f}; W&B run {run.url or run.id}")


def smoke_test(args: argparse.Namespace) -> None:
    """Dependency and shape check without BraTS data or a W&B run."""
    seed_everything(7)
    model = NNUNet((32, 32, 32), base_channels=4) if args.model == "nnunet" else BasicUNet(base_channels=4)
    image = torch.rand(1, 4, 32, 32, 32)
    label = torch.zeros(1, 1, 32, 32, 32, dtype=torch.int64)
    label[:, :, 3:12, 3:12, 3:12] = 1
    label[:, :, 5:10, 5:10, 5:10] = 2
    label[:, :, 6:9, 6:9, 6:9] = 3
    logits = model(image)
    assert logits.shape == (1, 4, 32, 32, 32)
    outputs = model(image, return_aux=True) if args.loss == "nnunet" else logits
    loss = (nnunet_loss if args.loss == "nnunet" else segmentation_loss)(outputs, label)
    loss.backward()
    model.eval()
    with torch.inference_mode():
        a = sliding_window_inference(image, (32, 32, 32), 1, model)
        b = sliding_window_inference(image, (32, 32, 32), 1, model)
    assert torch.allclose(a, b)
    print(f"Smoke test passed: logits {tuple(logits.shape)}, loss {loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Folder containing one directory per BraTS case")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--modality-suffixes", type=lambda s: tuple(s.split(",")), default=("t1", "t1ce", "t2", "flair"))
    parser.add_argument("--label-suffix", default="seg")
    parser.add_argument("--max-cases", type=int, help="Optional small subset for a local pipeline check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=lambda s: tuple(int(v) for v in s.split(",")), default=(64, 64, 64))
    parser.add_argument("--patches-per-case", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--model", choices=("basic", "nnunet"), default="basic")
    parser.add_argument("--loss", choices=("baseline", "nnunet"), default="baseline")
    parser.add_argument("--optimizer", choices=("adam", "nnunet"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--wandb-project", default="brats-nnunet-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and args.data_root is None:
        parser.error("--data-root is required unless --smoke-test is set")
    if len(args.patch_size) != 3 or any(value < 4 or value % 4 for value in args.patch_size):
        parser.error("--patch-size needs three dimensions, each at least 4 and divisible by 4")
    if min(args.epochs, args.val_interval, args.batch_size, args.patches_per_case, args.base_channels, args.sw_batch_size) < 1:
        parser.error("epochs, intervals, batch sizes, patches per case, and channels must be positive")
    if args.max_cases is not None and args.max_cases < 2:
        parser.error("--max-cases must be at least 2")
    if args.loss == "nnunet" and args.model != "nnunet":
        parser.error("--loss nnunet needs --model nnunet for the auxiliary output")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        smoke_test(arguments)
    else:
        run_training(arguments)
