
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Paths & hyperparameters (match Unet_Pet_Seg_Backbone_ResNet34_multiclass.ipynb) ──
ROOT = Path("/home/ahsan/test-project/ISIC-2019/Pet_data")
PET_DATA_ROOT = ROOT / "pet_fixed_by_label"
PET_CSV_PATH = (
    PET_DATA_ROOT
    / "Metadata/Metadata_final/Cat_black_grey_Dog_brown_white_train.csv"
)
TEST_CSV_PATH = (
    PET_DATA_ROOT
    / "Metadata/Metadata_final/Cat_black_grey_Dog_brown_white_test_IID.csv"
)
OUTPUT_DIR = ROOT / "output" / "multiseed_runs"
CHECKPOINT_DIR = ROOT / "checkpoints" / "multiseed"

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_CLASSES = 3
SPECIES_TO_CLASS = {"cat": 1, "dog": 2}
CLASS_NAMES = ["background", "cat", "dog"]
DASHBOARD_CLASSES = ["Background", "Cats", "Dogs"]
EPOCHS = 100
LR_INITIAL = 1e-4
VAL_SPLIT = 0.1
SEEDS = [22, 42, 123]
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_transform = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class PetDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None, indices=None):
        self.data_root = Path(data_root)
        self.transform = transform
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[list(indices)].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["image"]
        mask_path = self.data_root / row["mask"]

        image = np.array(Image.open(img_path).convert("RGB"))
        binary_mask = np.array(Image.open(mask_path).convert("L")) > 127
        species_class = SPECIES_TO_CLASS[str(row["species"]).lower()]
        mask = np.zeros(binary_mask.shape, dtype=np.int64)
        mask[binary_mask] = species_class

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        if mask.is_floating_point():
            mask = mask.round()
        return image, mask.long()


class PetSubgroupDataset(Dataset):
    def __init__(self, sub_df, data_root, transform=None):
        self.df = sub_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["image"]
        mask_path = self.data_root / row["mask"]
        species = str(row["species"]).lower().strip()

        image = np.array(Image.open(img_path).convert("RGB"))
        binary_mask = np.array(Image.open(mask_path).convert("L")) > 127
        species_cls = SPECIES_TO_CLASS[species]
        mask = np.zeros(binary_mask.shape, dtype=np.int64)
        mask[binary_mask] = species_cls

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        if mask.is_floating_point():
            mask = mask.round()
        return image, mask.long()


def compute_batch_metrics(outputs, masks, num_classes, smooth=1e-6):
    preds = outputs.argmax(dim=1)
    acc = (preds == masks).float().mean().item()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)
    preds_one_hot = F.one_hot(preds_flat, num_classes=num_classes).float()
    masks_one_hot = F.one_hot(masks_flat, num_classes=num_classes).float()

    inter = (preds_one_hot * masks_one_hot).sum(dim=0)
    total_preds = preds_one_hot.sum(dim=0)
    total_masks = masks_one_hot.sum(dim=0)
    union = total_preds + total_masks - inter

    inter = inter[1:]
    union = union[1:]
    total_preds = total_preds[1:]
    total_masks = total_masks[1:]
    class_present = total_masks > 0

    if class_present.sum() == 0:
        return {"acc": acc, "dice": 0.0, "iou": 0.0}

    dice_per_class = (2 * inter + smooth) / (total_preds + total_masks + smooth)
    iou_per_class = (inter + smooth) / (union + smooth)
    return {
        "acc": acc,
        "dice": dice_per_class[class_present].mean().item(),
        "iou": iou_per_class[class_present].mean().item(),
    }


def build_loaders(seed: int):
    pet_df = pd.read_csv(PET_CSV_PATH)
    train_indices, val_indices = train_test_split(
        pet_df.index,
        test_size=VAL_SPLIT,
        random_state=seed,
        stratify=pet_df["species"],
    )
    train_dataset = PetDataset(
        PET_CSV_PATH, PET_DATA_ROOT, transform=train_transform, indices=train_indices
    )
    val_dataset = PetDataset(
        PET_CSV_PATH, PET_DATA_ROOT, transform=val_transform, indices=val_indices
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_seed(seed: int) -> dict:
    set_seed(seed)
    train_loader, val_loader = build_loaders(seed)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": [],
    }

    print(f"\n{'=' * 70}\nTraining seed {seed} on {DEVICE}\n{'=' * 70}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_dice = train_acc = train_iou = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            m = compute_batch_metrics(outputs, masks, NUM_CLASSES)
            train_acc += m["acc"]
            train_dice += m["dice"]
            train_iou += m["iou"]

        model.eval()
        val_loss = val_dice = val_acc = val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                m = compute_batch_metrics(outputs, masks, NUM_CLASSES)
                val_acc += m["acc"]
                val_dice += m["dice"]
                val_iou += m["iou"]

        n_train, n_val = len(train_loader), len(val_loader)
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_val
        avg_train_dice = train_dice / n_train
        avg_val_dice = val_dice / n_val
        avg_train_acc = train_acc / n_train
        avg_val_acc = val_acc / n_val
        avg_train_iou = train_iou / n_train
        avg_val_iou = val_iou / n_val

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_dice"].append(avg_train_dice)
        history["val_dice"].append(avg_val_dice)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["train_iou"].append(avg_train_iou)
        history["val_iou"].append(avg_val_iou)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"train loss {avg_train_loss:.4f} dice {avg_train_dice:.4f} iou {avg_train_iou:.4f} | "
                f"val loss {avg_val_loss:.4f} dice {avg_val_dice:.4f} iou {avg_val_iou:.4f}"
            )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"UNet_multiclass_seed{seed}_epoch{EPOCHS}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")

    test_metrics = evaluate_on_test(model, criterion)
    return {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "final_val": {
            "loss": history["val_loss"][-1],
            "dice": history["val_dice"][-1],
            "acc": history["val_acc"][-1],
            "iou": history["val_iou"][-1],
        },
        "history": history,
        "test": test_metrics,
    }


def evaluate_on_test(model, criterion) -> dict:
    df_all = pd.read_csv(TEST_CSV_PATH)
    df_all.columns = df_all.columns.str.strip()
    df_all["species"] = df_all["species"].str.lower().str.strip()
    df_all["color"] = df_all["color"].str.lower().str.strip()

    subgroups = [
        {"species": sp, "color": col}
        for (sp, col) in df_all.groupby(["species", "color"]).groups.keys()
    ]

    model.eval()
    subgroup_results = {}
    for sg in subgroups:
        species, color = sg["species"], sg["color"]
        key = f"{species}_{color}"
        label = f"{species.capitalize()}/{color.capitalize()}"
        sub_df = df_all[(df_all["species"] == species) & (df_all["color"] == color)]
        loader = DataLoader(
            PetSubgroupDataset(sub_df, PET_DATA_ROOT, transform=val_transform),
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        tot_acc = tot_dice = tot_iou = tot_loss = 0.0
        with torch.no_grad():
            for images, masks in loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                tot_loss += criterion(outputs, masks).item()
                m = compute_batch_metrics(outputs, masks, NUM_CLASSES)
                tot_acc += m["acc"]
                tot_dice += m["dice"]
                tot_iou += m["iou"]

        n = len(loader)
        subgroup_results[key] = {
            "label": label,
            "n": len(sub_df),
            "loss": tot_loss / n,
            "acc": tot_acc / n,
            "dice": tot_dice / n,
            "iou": tot_iou / n,
        }

    full_loader = DataLoader(
        PetSubgroupDataset(df_all, PET_DATA_ROOT, transform=val_transform),
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in full_loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu().numpy().flatten()
            all_preds.append(preds)
            all_targets.append(masks.numpy().flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class_iou = {}
    for i, cls in enumerate(DASHBOARD_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        denom = tp + fp + fn
        per_class_iou[cls] = float(tp / denom) if denom > 0 else 0.0

    pixel_acc = float(np.diag(cm).sum() / cm.sum())
    mean_iou = float(np.mean(list(per_class_iou.values())))

    overall_fg = np.mean(
        [subgroup_results[k]["iou"] for k in subgroup_results]
    )

    return {
        "subgroups": subgroup_results,
        "full_test": {
            "pixel_acc": pixel_acc,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou,
        },
        "overall_subgroup_mean_iou": float(overall_fg),
    }


def mean_std(values: list[float]) -> str:
    arr = np.array(values, dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def print_summary(all_runs: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print("MULTI-SEED SUMMARY (mean ± std across 3 runs)")
    print(f"{'=' * 70}")

    print("\n── Final validation (end of training) ──")
    for metric in ["loss", "dice", "acc", "iou"]:
        vals = [r["final_val"][metric] for r in all_runs]
        print(f"  Val {metric:5s}: {mean_std(vals)}")

    print("\n── Full IID test set (pixel-level) ──")
    for cls in DASHBOARD_CLASSES:
        vals = [r["test"]["full_test"]["per_class_iou"][cls] for r in all_runs]
        print(f"  IoU [{cls:12s}]: {mean_std(vals)}")
    for metric in ["pixel_acc", "mean_iou"]:
        vals = [r["test"]["full_test"][metric] for r in all_runs]
        print(f"  {metric:12s}: {mean_std(vals)}")

    print("\n── Per-subgroup test IoU ──")
    subgroup_keys = list(all_runs[0]["test"]["subgroups"].keys())
    for key in subgroup_keys:
        label = all_runs[0]["test"]["subgroups"][key]["label"]
        vals = [r["test"]["subgroups"][key]["iou"] for r in all_runs]
        print(f"  {label:16s}: {mean_std(vals)}")

    print("\n── Per-run full-test mean IoU ──")
    for r in all_runs:
        miou = r["test"]["full_test"]["mean_iou"]
        print(f"  seed {r['seed']:3d}: {miou:.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = []
    for seed in SEEDS:
        result = train_one_seed(seed)
        all_runs.append(result)
        run_path = OUTPUT_DIR / f"run_seed{seed}.json"
        with open(run_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote per-run metrics → {run_path}")

    summary = {
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "runs": all_runs,
    }
    summary_path = OUTPUT_DIR / "multiseed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_runs)
    print(f"\nFull results saved → {summary_path}")


if __name__ == "__main__":
    main()
