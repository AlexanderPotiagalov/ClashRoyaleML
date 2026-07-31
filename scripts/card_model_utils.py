from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str((Path("models/.cache/torch")).resolve()))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


VISUAL_CLASSES = [
    "electro_spirit", "fireball", "barbarian_barrel", "knight",
    "hero_musketeer", "hog_rider", "cannon", "cannon_evolution",
    "skeletons", "empty",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = (160, 112)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def logical_fields(visual_label: str) -> tuple[str, bool]:
    if visual_label == "cannon_evolution":
        return "cannon", True
    return visual_label, False


def build_transform(training: bool = False):
    operations = []
    if training:
        operations.extend([
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.RandomAffine(
                degrees=0, translate=(0.025, 0.025), scale=(0.96, 1.04),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ])
    operations.extend([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)


class CardManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, classes: list[str], training: bool = False):
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Manifest contains no {split!r} samples: {manifest}")
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.transform = build_transform(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[row["visual_label"]], index


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model_bundle(checkpoint_path: Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = create_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, classes, checkpoint


def prediction_fields(visual_label: str, confidence: float, threshold: float) -> dict[str, object]:
    if confidence < threshold:
        return {
            "visual_label": "unknown", "logical_card": "unknown",
            "is_evolved": False, "confidence": confidence,
        }
    logical_card, is_evolved = logical_fields(visual_label)
    return {
        "visual_label": visual_label, "logical_card": logical_card,
        "is_evolved": is_evolved, "confidence": confidence,
    }


def save_classes(path: Path, classes: list[str]) -> None:
    path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
