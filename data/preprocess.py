"""
SpikeGuard — Wildlife Camera-Trap Dataset Pipeline

Supports:
1. iWildCam 2020 mini-subset download from public sources
2. Fallback simulation mode using CIFAR-10 remapped to wildlife classes
3. Standard preprocessing for SNN rate-coding input

For hackathon: use simulation mode for quick iteration, real data for final eval.
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
import torchvision

from utils import WILDLIFE_CLASSES, NUM_CLASSES


# ─── Transforms ──────────────────────────────────────────────
IMG_SIZE = 64  # Small size for fast SNN training

TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomCrop(IMG_SIZE, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DENORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor):
    """Denormalize image tensor for display."""
    return tensor * DENORM_STD.to(tensor.device) + DENORM_MEAN.to(tensor.device)


# ─── Simulation Dataset (CIFAR-10 → Wildlife Classes) ────────
class WildlifeSimDataset(Dataset):
    """
    Simulates camera-trap data by remapping CIFAR-10 classes to wildlife categories.

    Mapping:  airplane→blank, automobile→blank, bird→bird, cat→leopard,
              deer→deer, dog→wild_boar, frog→monkey, horse→elephant,
              ship→blank, truck→human

    This is for QUICK PROTOTYPING ONLY. For real evaluation, use iWildCam.
    """

    # CIFAR-10 class → Wildlife class index
    REMAP = {
        0: 9,  # airplane → blank
        1: 9,  # automobile → blank
        2: 7,  # bird → bird
        3: 1,  # cat → leopard
        4: 3,  # deer → deer
        5: 4,  # dog → wild_boar
        6: 6,  # frog → monkey
        7: 2,  # horse → elephant
        8: 9,  # ship → blank
        9: 8,  # truck → human
    }

    def __init__(self, root="./data/cifar10", train=True, transform=None):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True,
        )
        self.transform = transform or (TRAIN_TRANSFORM if train else TEST_TRANSFORM)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]
        # img is PIL Image
        wildlife_label = self.REMAP[label]
        img = self.transform(img)
        return img, wildlife_label


# ─── Real Wildlife Dataset (folder-based) ────────────────────
class WildlifeImageDataset(Dataset):
    """
    Loads wildlife images from a folder structure:
        data_dir/
            tiger/
                img001.jpg
            leopard/
                img002.jpg
            ...

    For iWildCam or custom datasets organized by class folder.
    """

    def __init__(self, data_dir, transform=None, max_per_class=None):
        self.data_dir = data_dir
        self.transform = transform or TEST_TRANSFORM
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(WILDLIFE_CLASSES)}

        for cls_name in WILDLIFE_CLASSES:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(os.listdir(cls_dir))
            if max_per_class:
                files = files[:max_per_class]
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name],
                    ))

        print(f"  WildlifeImageDataset: {len(self.samples)} images "
              f"from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ─── DataLoader Factory ──────────────────────────────────────
def get_dataloaders(mode="simulation", data_dir="./data/wildlife",
                    batch_size=64, val_split=0.15, test_split=0.15,
                    num_workers=2, max_per_class=None):
    """
    Create train, val, test dataloaders.

    Args:
        mode: "simulation" (CIFAR-10 remap) or "real" (folder-based).
        data_dir: Data directory.
        batch_size: Batch size.
        val_split: Validation fraction.
        test_split: Test fraction.
        num_workers: DataLoader workers.
        max_per_class: Max images per class (for quick experiments).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if mode == "simulation":
        train_dataset = WildlifeSimDataset(train=True, transform=TRAIN_TRANSFORM)
        test_dataset = WildlifeSimDataset(train=False, transform=TEST_TRANSFORM)

        # Split training into train + val
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"  Simulation mode: {train_size} train, {val_size} val, "
              f"{len(test_dataset)} test")

    elif mode == "real":
        dataset = WildlifeImageDataset(
            data_dir, transform=TRAIN_TRANSFORM, max_per_class=max_per_class,
        )
        n = len(dataset)
        test_n = int(n * test_split)
        val_n = int(n * val_split)
        train_n = n - val_n - test_n

        train_subset, val_subset, test_dataset = random_split(
            dataset, [train_n, val_n, test_n],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"  Real data mode: {train_n} train, {val_n} val, {test_n} test")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
