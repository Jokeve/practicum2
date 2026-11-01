"""Small synthetic BraTS cases exercise file loading and full-volume inference."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader

from main import list_cases, split_cases
from src.data_loader import advanced_train_transforms, val_transforms
from src.evaluation import dice_per_class, evaluate
from src.loss import segmentation_loss
from src.model import BasicUNet


def write_case(root: Path, case_id: str) -> None:
    folder = root / case_id
    folder.mkdir()
    shape = (16, 16, 16)
    mask = np.zeros(shape, dtype=np.int16)
    mask[3:12, 3:12, 3:12] = 2
    mask[5:10, 5:10, 5:10] = 1
    mask[6:9, 6:9, 6:9] = 3
    for index, modality in enumerate(("t1", "t1ce", "t2", "flair")):
        volume = np.full(shape, index + 1, dtype=np.float32)
        volume[mask > 0] += 1
        nib.save(nib.Nifti1Image(volume, np.eye(4)), folder / f"{case_id}_{modality}.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), folder / f"{case_id}_seg.nii.gz")


def write_msd_case(root: Path, case_id: str) -> None:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    volume = np.ones((16, 16, 16, 4), dtype=np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.int16)
    mask[3:12, 3:12, 3:12] = 1
    mask[5:10, 5:10, 5:10] = 2
    mask[6:9, 6:9, 6:9] = 3
    nib.save(nib.Nifti1Image(volume, np.eye(4)), images / f"{case_id}.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), labels / f"{case_id}.nii.gz")


def test_data_model_loss_and_full_volume_validation(tmp_path: Path) -> None:
    write_case(tmp_path, "Case_001")
    write_case(tmp_path, "Case_002")
    cases = list_cases(tmp_path, ("t1", "t1ce", "t2", "flair"), "seg")
    train, val = split_cases(cases)
    assert {case["case_id"] for case in train}.isdisjoint({case["case_id"] for case in val})
    train_set = CacheDataset(train, advanced_train_transforms((16, 16, 16), 1), cache_num=1, num_workers=0, progress=False)
    train_batch = next(iter(DataLoader(train_set, batch_size=1)))
    image, label = train_batch["image"], train_batch["label"]
    assert image.shape == (1, 4, 16, 16, 16)
    assert label.shape == (1, 1, 16, 16, 16)
    assert set(torch.unique(label).tolist()) == {0, 1, 2, 3}

    model = BasicUNet(base_channels=4)
    logits = model(image)
    loss = segmentation_loss(logits, label)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None
    perfect_logits = torch.nn.functional.one_hot(label[:, 0], num_classes=4).movedim(-1, 1).float() * 20
    assert dice_per_class(perfect_logits, label) == [1.0, 1.0, 1.0]

    val_set = CacheDataset(val, val_transforms(), cache_num=1, num_workers=0, progress=False)
    val_loader = DataLoader(val_set, batch_size=1)
    first, rows = evaluate(model, val_loader, torch.device("cpu"), (16, 16, 16), 1)
    second, _ = evaluate(model, val_loader, torch.device("cpu"), (16, 16, 16), 1)
    assert first == second
    assert len(rows) == 1 and set(first) == {"edema", "non_enhancing_tumor", "enhancing_tumor", "mean"}


def test_validation_split_is_fixed_and_independent_of_case_order() -> None:
    cases = [{"case_id": f"BRATS_{index:03d}"} for index in range(10)]
    train, val = split_cases(cases)
    train_reversed, val_reversed = split_cases(list(reversed(cases)))
    assert len(train) == 8 and len(val) == 2
    assert {case["case_id"] for case in train}.isdisjoint({case["case_id"] for case in val})
    assert [case["case_id"] for case in val] == [case["case_id"] for case in val_reversed]
    assert [case["case_id"] for case in train] == [case["case_id"] for case in train_reversed]


def test_msd_four_dimensional_layout(tmp_path: Path) -> None:
    write_msd_case(tmp_path, "BRATS_001")
    cases = list_cases(tmp_path, ("t1", "t1ce", "t2", "flair"), "seg")
    sample = CacheDataset(cases, val_transforms(), cache_num=1, num_workers=0, progress=False)[0]
    assert sample["image"].shape == (4, 16, 16, 16)
    assert sample["label"].shape == (1, 16, 16, 16)
    assert (sample["label"] == 1).sum().item() == 604
    assert (sample["label"] == 2).sum().item() == 98
    assert (sample["label"] == 3).sum().item() == 27
