from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path("models/.cache/torch").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("models/.cache/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from card_model_utils import CardManifestDataset, VISUAL_CLASSES, create_model, save_classes
except ModuleNotFoundError:
    from scripts.card_model_utils import CardManifestDataset, VISUAL_CLASSES, create_model, save_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3-Small for card-slot classification.")
    parser.add_argument("--manifest", type=Path, default=Path("data/card_training/card_training_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/card_classifier_v1"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, true, predicted = [], [], []
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward(); optimizer.step()
        losses.append(float(loss.detach()) * len(targets))
        true.extend(targets.cpu().tolist())
        predicted.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": sum(losses) / len(true), "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=list(range(len(VISUAL_CLASSES))),
                             average="macro", zero_division=0),
    }


def save_checkpoint(path, model, optimizer, epoch, classes, history, best_f1):
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "classes": classes, "history": history,
        "best_validation_macro_f1": best_f1,
    }, path)


def plot_history(history, output):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="validation")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [r["train_macro_f1"] for r in history], label="train")
    axes[1].plot(epochs, [r["validation_macro_f1"] for r in history], label="validation")
    axes[1].set_title("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=.25)
    figure.tight_layout(); figure.savefig(output, dpi=160); plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = CardManifestDataset(args.manifest, "train", VISUAL_CLASSES, training=True)
    validation_data = CardManifestDataset(args.manifest, "validation", VISUAL_CLASSES)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda")
    class_counts = Counter(row["visual_label"] for row in train_data.rows)
    weights = torch.tensor([
        len(train_data) / (len(VISUAL_CLASSES) * class_counts[label]) for label in VISUAL_CLASSES
    ], dtype=torch.float32, device=device)
    model = create_model(len(VISUAL_CLASSES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=.4, patience=1)
    args.output.mkdir(parents=True, exist_ok=True)
    save_classes(args.output / "classes.json", VISUAL_CLASSES)
    history, best_f1, stale = [], -1.0, 0
    print(f"Device: {device}; train={len(train_data)} validation={len(validation_data)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_pass(model, train_loader, criterion, device, optimizer)
        validation_metrics = epoch_pass(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["macro_f1"])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"validation_{k}": v for k, v in validation_metrics.items()},
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = validation_metrics["macro_f1"] > best_f1 + 1e-5
        if improved:
            best_f1, stale = validation_metrics["macro_f1"], 0
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, VISUAL_CLASSES, history, best_f1)
        else:
            stale += 1
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, VISUAL_CLASSES, history, best_f1)
        (args.output / "training_history.json").write_text(json.dumps({
            "device": str(device), "best_validation_macro_f1": best_f1,
            "best_epoch": max(history, key=lambda r: r["validation_macro_f1"])["epoch"],
            "class_counts": class_counts, "epochs": history,
        }, indent=2), encoding="utf-8")
        plot_history(history, args.output / "training_curves.png")
        print(f"epoch={epoch} train_f1={train_metrics['macro_f1']:.4f} "
              f"val_f1={validation_metrics['macro_f1']:.4f} val_acc={validation_metrics['accuracy']:.4f}")
        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
