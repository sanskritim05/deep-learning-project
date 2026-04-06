from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
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
    total_samples = sum(class_counts.values())
    return torch.tensor(
        [total_samples / class_counts[i] for i in range(num_classes)],
        dtype=torch.float32,
    )


def _make_split_indices(dataset_size: int, val_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int((1.0 - val_ratio) * dataset_size)
    return indices[:train_size], indices[train_size:]


def _build_common_metadata(base_train_dataset, train_indices, val_indices, class_weights):
    return {
        "idx_to_class": {idx: name for idx, name in enumerate(base_train_dataset.classes)},
        "class_to_idx": base_train_dataset.class_to_idx,
        "class_weights": class_weights,
        "num_classes": len(base_train_dataset.classes),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "class_counts": Counter(base_train_dataset.targets),
    }


def prepare_cnn_dataloaders(
    train_dir: str | Path = "archive/train",
    test_dir: str | Path = "archive/test",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    image_size: int = 48,
    seed: int = 42,
):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    base_train_dataset = datasets.ImageFolder(train_dir)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_indices, val_indices = _make_split_indices(len(base_train_dataset), val_ratio, seed)
    train_dataset = TransformSubset(base_train_dataset, train_indices, transform=train_transform)
    val_dataset = TransformSubset(base_train_dataset, val_indices, transform=eval_transform)

    class_weights = build_class_weights(base_train_dataset.targets, len(base_train_dataset.classes))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = _build_common_metadata(base_train_dataset, train_indices, val_indices, class_weights)
    metadata.update(
        {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }
    )
    return metadata


def prepare_transfer_dataloaders(
    train_dir: str | Path = "archive/train",
    test_dir: str | Path = "archive/test",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    base_train_dataset = datasets.ImageFolder(train_dir)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_indices, val_indices = _make_split_indices(len(base_train_dataset), val_ratio, seed)
    train_dataset = TransformSubset(base_train_dataset, train_indices, transform=train_transform)
    val_dataset = TransformSubset(base_train_dataset, val_indices, transform=eval_transform)

    class_weights = build_class_weights(base_train_dataset.targets, len(base_train_dataset.classes))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    metadata = _build_common_metadata(base_train_dataset, train_indices, val_indices, class_weights)
    metadata.update(
        {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }
    )
    return metadata


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18_transfer_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    local_weights_path: str | Path | None = "model_weights/resnet18-f37072fd.pth",
):
    model = resnet18(weights=None)
    loaded_pretrained = False

    if pretrained:
        local_path = Path(local_weights_path) if local_weights_path else None
        if local_path and local_path.exists():
            state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            loaded_pretrained = True
            print(f"Loaded pretrained weights from local file: {local_path}")
        else:
            weights = ResNet18_Weights.DEFAULT
            try:
                model = resnet18(weights=weights)
                loaded_pretrained = True
            except Exception as exc:
                print(f"Could not load pretrained weights, falling back to random init: {exc}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model, loaded_pretrained


def unfreeze_resnet_stage4(model) -> None:
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True


def unfreeze_resnet_stage3_and_4(model) -> None:
    for name, param in model.named_parameters():
        if (
            name.startswith("layer3")
            or name.startswith("layer4")
            or name.startswith("fc")
        ):
            param.requires_grad = True


def build_weighted_loss(class_weights: torch.Tensor, device: torch.device, label_smoothing: float = 0.0):
    return nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )


def evaluate_model(model, loader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    patience: int | None = None,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if patience is not None and stale_epochs >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

    model.load_state_dict(best_state)
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    return history


def predict_with_confidence(model, loader, idx_to_class, device: torch.device, max_examples: int = 5):
    model.eval()
    shown = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            for i in range(images.size(0)):
                print(
                    f"True: {idx_to_class[labels[i].item()]} | "
                    f"Predicted: {idx_to_class[preds[i].item()]} | "
                    f"Confidence: {confidences[i].item():.4f}"
                )
                shown += 1
                if shown >= max_examples:
                    return probs

    return probs
