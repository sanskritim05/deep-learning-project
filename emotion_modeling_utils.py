"""emotion_modeling_utils_v2.py
-----------------
1. EfficientNet-B2 backbone  – stronger feature extractor than ResNet18
2. Richer data augmentation  – TrivialAugmentWide + RandomErasing + sharper crop/flip
3. MixUp training            – mixes two random samples per batch to smooth the decision boundary
4. Label smoothing = 0.1     – reduces over-confidence on a noisy emotion dataset
5. Cosine Annealing LR       – smoother decay than ReduceLROnPlateau; avoids stalling
6. Gradient clipping         – prevents exploding gradients during fine-tuning
7. Test-Time Augmentation    – averages 5 forward passes with random flips/crops at inference
8. Staged unfreezing         – head → top block → all blocks, each with its own LR
"""

from __future__ import annotations

import copy
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

# ImageNet stats (EfficientNet was trained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# Reproducibility
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Dataset helpers
class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_class_weights(targets, num_classes: int) -> torch.Tensor:
    class_counts = Counter(targets)
    total        = sum(class_counts.values())
    return torch.tensor(
        [total / class_counts[i] for i in range(num_classes)],
        dtype=torch.float32,
    )


def _make_split_indices(dataset_size: int, val_ratio: float, seed: int):
    generator  = torch.Generator().manual_seed(seed)
    indices    = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int((1.0 - val_ratio) * dataset_size)
    return indices[:train_size], indices[train_size:]


def _build_common_metadata(base_train, train_idx, val_idx, class_weights):
    return {
        "idx_to_class":  {i: n for i, n in enumerate(base_train.classes)},
        "class_to_idx":  base_train.class_to_idx,
        "class_weights": class_weights,
        "num_classes":   len(base_train.classes),
        "train_size":    len(train_idx),
        "val_size":      len(val_idx),
        "class_counts":  Counter(base_train.targets),
    }


# Stronger augmentation pipeline
def prepare_efficientnet_dataloaders(
    train_dir:  str | Path = "archive/train",
    test_dir:   str | Path = "archive/test",
    batch_size: int        = 32,          # EfficientNet-B2 needs more memory
    val_ratio:  float      = 0.2,
    image_size: int        = 260,         # native resolution for B2
    seed:       int        = 42,
):
    """
    Dataloaders tuned for EfficientNet-B2.

    Training augmentations (change 2):
    - TrivialAugmentWide  : randomly applies one of ~30 distortions per batch
    - RandomHorizontalFlip: faces are roughly symmetric
    - RandomResizedCrop   : forces the model to use partial face cues
    - ColorJitter         : handles lighting variation
    - RandomErasing       : randomly masks a patch → forces learning from context
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),          # grayscale → 3-ch copy
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),                       # NEW
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # STRONGER
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    base_train = datasets.ImageFolder(train_dir)
    test_ds    = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_idx, val_idx = _make_split_indices(len(base_train), val_ratio, seed)
    train_ds = TransformSubset(base_train, train_idx, transform=train_transform)
    val_ds   = TransformSubset(base_train, val_idx,   transform=eval_transform)

    class_weights = build_class_weights(base_train.targets, len(base_train.classes))

    # pin_memory speeds up CPU→GPU transfers on CUDA.
    # persistent_workers avoids restarting worker processes each epoch.
    # dynamic num_workers — MPS is unstable with many workers; CUDA benefits from more.
    pin_memory = torch.cuda.is_available()
    num_workers = 0 if torch.backends.mps.is_available() else min(8, os.cpu_count() or 4)
    persistent = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent)

    meta = _build_common_metadata(base_train, train_idx, val_idx, class_weights)
    meta.update({"train_loader": train_loader,
                 "val_loader":   val_loader,
                 "test_loader":  test_loader})
    return meta


# Change 1: EfficientNet-B2 model builder
def build_efficientnet_model(
    num_classes:     int,
    pretrained:      bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    EfficientNet-B2 is roughly 3× more parameter-efficient than ResNet18 while
    achieving significantly higher ImageNet top-1 accuracy.  Its compound scaling
    (width + depth + resolution) makes it well-suited to fine-grained tasks like
    facial expression recognition.
    """
    weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
    model   = efficientnet_b2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),                       # smoother than ReLU for EfficientNet
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    # Always keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_top_blocks(model: nn.Module, num_blocks: int = 3) -> None:
    """Unfreeze the last `num_blocks` feature blocks + classifier."""
    features = list(model.features.children())
    for block in features[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# MixUp helper
def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.4,
):
    """
    MixUp: creates a convex combination of two random samples.
    The model is trained on blended images with blended soft labels.
    This strongly regularises the decision boundary between similar emotions
    (e.g. fear vs surprise) and consistently improves generalisation.

    Returns (mixed_images, soft_labels_a, soft_labels_b, lam)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index      = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]

    # Convert to one-hot for soft mixing
    one_hot_a = torch.zeros(batch_size, num_classes, device=images.device)
    one_hot_b = torch.zeros(batch_size, num_classes, device=images.device)
    one_hot_a.scatter_(1, labels.unsqueeze(1), 1)
    one_hot_b.scatter_(1, labels[index].unsqueeze(1), 1)

    return mixed_images, one_hot_a, one_hot_b, lam


# Loss
def build_weighted_loss(
    class_weights:   torch.Tensor,
    device:          torch.device,
    label_smoothing: float = 0.1,       #increased from 0.05
) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )


# Evaluation
def evaluate_model(model, loader, criterion, device: torch.device):
    model.eval()
    running_loss = correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs         = model(images)
            loss            = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct      += (outputs.argmax(1) == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# Test-Time Augmentation
def evaluate_with_tta(
    model,
    loader,
    criterion,
    device:    torch.device,
    tta_n:     int   = 5,
    image_size: int  = 260,
):
    """
    FIX 4: TTA augmentation now runs entirely on GPU using torchvision functional ops
    instead of a serial per-image CPU loop. This eliminates the expensive
    `torch.stack([tta_transform(img.cpu()) for img in images])` bottleneck.

    NOTE: Only call this function at final test time — NOT during per-epoch validation.
    Use evaluate_model() for validation inside the training loop.
    """
    import torchvision.transforms.functional as TF

    model.eval()
    running_loss = correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            probs = torch.zeros(images.size(0), model.classifier[-1].out_features,
                                device=device)
            for _ in range(tta_n):
                # horizontal flip applied on-GPU as a tensor op (no CPU roundtrip)
                aug = images.clone()
                if torch.rand(1).item() > 0.5:
                    aug = TF.hflip(aug)

                # random crop via affine (stays on GPU)
                scale = torch.empty(1).uniform_(0.85, 1.0).item()
                crop_size = int(image_size * scale)
                aug = TF.center_crop(aug, crop_size)
                aug = TF.resize(aug, [image_size, image_size], antialias=True)

                probs += torch.softmax(model(aug), dim=1)

            probs /= tta_n
            preds  = probs.argmax(dim=1)

            loss         = criterion(model(images), labels)
            running_loss += loss.item() * labels.size(0)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# Training loop with cosine LR + gradient clipping
def train_model_v2(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device:        torch.device,
    epochs:        int,
    num_classes:   int,
    scheduler      = None,
    patience:      int | None = None,
    use_mixup:     bool       = True,
    mixup_alpha:   float      = 0.4,
    grad_clip:     float      = 1.0,
):
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_state   = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch   = 0
    stale        = 0

    # Automatic Mixed Precision - halves memory bandwidth, uses Tensor Cores on CUDA.
    # GradScaler is a no-op on CPU/MPS so this is safe across all device types.
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        run_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # zero_grad(set_to_none=True) skips the memset to zero,
            # which is faster than the default zero_grad().
            optimizer.zero_grad(set_to_none=True)

            # wrap forward pass in autocast for AMP
            with autocast(enabled=use_amp):
                if use_mixup:
                    mixed, oh_a, oh_b, lam = mixup_batch(images, labels, num_classes, mixup_alpha)
                    outputs  = model(mixed)
                    log_prob = torch.log_softmax(outputs, dim=1)
                    loss     = -(lam * (oh_a * log_prob).sum(1) +
                                 (1 - lam) * (oh_b * log_prob).sum(1)).mean()
                else:
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

            # scale loss, unscale before grad clip, then step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * labels.size(0)
            correct  += (outputs.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct  / total

        # Use evaluate_model (single forward pass) for per-epoch validation.
        # evaluate_with_tta is expensive (5× forward passes + augmentation) and should
        # only be called ONCE on the test set after training completes.
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(
            ("train_loss", "train_acc", "val_loss", "val_acc", "lr"),
            (train_loss,   train_acc,   val_loss,   val_acc,   lr),
        ):
            history[k].append(v)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            best_state   = copy.deepcopy(model.state_dict())
            stale        = 0
        else:
            stale += 1

        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {lr:.2e}"
        )

        if patience and stale >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_state)
    history["best_val_acc"] = best_val_acc
    history["best_epoch"]   = best_epoch
    print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    return history


def predict_with_confidence(model, loader, idx_to_class, device, max_examples=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs          = torch.softmax(model(images), dim=1)
            confs, preds   = probs.max(dim=1)
            for i in range(images.size(0)):
                print(
                    f"True: {idx_to_class[labels[i].item()]:8s} | "
                    f"Predicted: {idx_to_class[preds[i].item()]:8s} | "
                    f"Confidence: {confs[i].item():.4f}"
                )
                shown += 1
                if shown >= max_examples:
                    return probs
    return probs
