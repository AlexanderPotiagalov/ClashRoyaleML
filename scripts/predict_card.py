from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

try:
    from card_model_utils import IMAGE_EXTENSIONS, build_transform, load_model_bundle, prediction_fields
except ModuleNotFoundError:
    from scripts.card_model_utils import IMAGE_EXTENSIONS, build_transform, load_model_bundle, prediction_fields


def parse_args():
    parser = argparse.ArgumentParser(description="Predict card identities for an image or directory.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/card_classifier_v1/best.pt"))
    parser.add_argument("--confidence-threshold", type=float, default=.65)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def discover_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(path)


def predict_paths(model, classes, paths, threshold, device):
    transform = build_transform(False)
    results = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probabilities = model(tensor).softmax(1)[0]
            confidence, index = probabilities.max(0)
            fields = prediction_fields(classes[int(index)], float(confidence), threshold)
            results.append({"image_path": str(path.resolve()), **fields})
    return results


def main() -> int:
    args = parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, _ = load_model_bundle(args.model, device)
    results = predict_paths(model, classes, discover_images(args.input), args.confidence_threshold, device)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "visual_label", "logical_card", "is_evolved", "confidence"])
            writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
