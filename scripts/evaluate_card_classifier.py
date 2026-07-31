from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path("models/.cache/torch").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("models/.cache/matplotlib").resolve()))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

try:
    from card_model_utils import CardManifestDataset, load_model_bundle
except ModuleNotFoundError:
    from scripts.card_model_utils import CardManifestDataset, load_model_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the neural card classifier on held-out matches.")
    parser.add_argument("--manifest", type=Path, default=Path("data/card_training/card_training_manifest.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/card_classifier_v1/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/card_classifier_v1/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--incorrect-limit", type=int, default=48)
    return parser.parse_args()


def plot_confusion(matrix, classes, output):
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title("Held-out test confusion matrix")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output, dpi=170); plt.close(figure)


def incorrect_sheet(rows, output, limit):
    selected = rows[:limit]; columns, width, height = 6, 170, 210
    sheet = np.full((max(1, int(np.ceil(len(selected) / columns))) * height, columns * width, 3), 28, np.uint8)
    for index, row in enumerate(selected):
        image = cv2.imread(row["image_path"])
        if image is None: continue
        scale = min(150 / image.shape[1], 150 / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grid_row, column = divmod(index, columns); x = column * width + (width-image.shape[1])//2; y = grid_row*height
        sheet[y:y+image.shape[0], x:x+image.shape[1]] = image
        cv2.putText(sheet, f"T:{row['visual_label']}"[:25], (column*width+3, y+168), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
        cv2.putText(sheet, f"P:{row['predicted_visual_label']} {float(row['confidence']):.2f}"[:26], (column*width+3, y+187), cv2.FONT_HERSHEY_SIMPLEX, .35, (120,210,255), 1)
    if not cv2.imwrite(str(output), sheet): raise RuntimeError(f"Could not write {output}")


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, checkpoint = load_model_bundle(args.model, device)
    dataset = CardManifestDataset(args.manifest, "test", classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    true, predicted, confidences, sample_indices = [], [], [], []
    with torch.inference_mode():
        for images, targets, indices in loader:
            probabilities = model(images.to(device)).softmax(1).cpu()
            confidence, prediction = probabilities.max(1)
            true.extend(targets.tolist()); predicted.extend(prediction.tolist())
            confidences.extend(confidence.tolist()); sample_indices.extend(indices.tolist())
    labels = list(range(len(classes))); matrix = confusion_matrix(true, predicted, labels=labels)
    report = classification_report(true, predicted, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    cannon_indices = [index for index, target in enumerate(true) if classes[target] in {"cannon", "cannon_evolution"}]
    cannon_correct = sum(predicted[i] == true[i] for i in cannon_indices)
    metrics = {
        "checkpoint": str(args.model), "checkpoint_epoch": checkpoint.get("epoch"),
        "test_matches": sorted({row["match_id"] for row in dataset.rows}),
        "sample_count": len(true), "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=labels, average="macro", zero_division=0),
        "per_class": {name: report[name] for name in classes},
        "cannon_vs_cannon_evolution": {
            "sample_count": len(cannon_indices),
            "accuracy": cannon_correct / len(cannon_indices) if cannon_indices else None,
            "cannon_accuracy": report["cannon"]["recall"],
            "cannon_evolution_accuracy": report["cannon_evolution"]["recall"],
        },
        "appearance_breakdown": {
            "available": False,
            "reason": "The source manifests do not contain reviewed normal/greyed appearance metadata."
        },
        "confusion_matrix": matrix.tolist(), "classes": classes,
    }
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_rows = []
    for target, prediction, confidence, sample_index in zip(true, predicted, confidences, sample_indices):
        source = dataset.rows[sample_index]
        prediction_rows.append({**source, "predicted_visual_label": classes[prediction],
                                "confidence": confidence, "correct": target == prediction})
    fields = list(prediction_rows[0])
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)
    plot_confusion(matrix, classes, args.output / "confusion_matrix.png")
    incorrect_sheet([row for row in prediction_rows if not row["correct"]],
                    args.output / "incorrect_predictions.jpg", args.incorrect_limit)
    print(json.dumps({k: metrics[k] for k in ("test_matches", "sample_count", "accuracy", "macro_f1", "cannon_vs_cannon_evolution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
