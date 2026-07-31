from __future__ import annotations

import argparse
import csv
import hashlib
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
    from card_model_utils import CardManifestDataset, load_model_bundle, logical_fields
    from prepare_card_training_data import read_match
except ModuleNotFoundError:
    from scripts.card_model_utils import CardManifestDataset, load_model_bundle, logical_fields
    from scripts.prepare_card_training_data import read_match


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a frozen card model on untouched external matches.")
    parser.add_argument("--matches", nargs="+", default=["match_007", "match_008", "match_009"])
    parser.add_argument("--model", type=Path, default=Path("models/card_classifier_v1/best.pt"))
    parser.add_argument("--split-config", type=Path, default=Path("config/card_training_split.json"))
    parser.add_argument("--training-manifest", type=Path, default=Path("data/card_training/card_training_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/card_dataset"))
    parser.add_argument("--classification-root", type=Path, default=Path("data/card_classifications"))
    parser.add_argument("--output", type=Path, default=Path("models/card_classifier_v1/external_evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sheet-limit", type=int, default=48)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provenance(matches, split_config, training_manifest, checkpoint):
    split = json.loads(split_config.read_text(encoding="utf-8"))
    configured = set(split["train_matches"] + split["validation_matches"] + split["test_matches"])
    with training_manifest.open(newline="", encoding="utf-8") as handle:
        manifest_matches = {row["match_id"] for row in csv.DictReader(handle)}
    external = set(matches)
    checks = {
        "not_in_training_split": external.isdisjoint(split["train_matches"]),
        "not_in_validation_or_model_selection": external.isdisjoint(split["validation_matches"]),
        "not_in_original_test": external.isdisjoint(split["test_matches"]),
        "not_in_training_manifest": external.isdisjoint(manifest_matches),
        "not_in_any_saved_split": external.isdisjoint(configured),
        "prototype_creation_not_applicable": True,
        "threshold_tuning_not_used": True,
    }
    if not all(checks.values()):
        raise ValueError(f"External-evaluation leakage detected: {checks}")
    return {
        "verified": True, "checks": checks,
        "explanation": {
            "prototype_creation": "The frozen MobileNet checkpoint performs neural inference and contains no prototype bank.",
            "threshold_tuning": "Metrics use raw argmax predictions; confidence thresholds are reported post hoc and do not change predictions.",
            "model_selection": f"Checkpoint selection used validation matches {split['validation_matches']} only.",
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
        "saved_split": split,
    }


def write_manifest(rows, path):
    fields = ["image_path", "match_id", "slot", "visual_label", "logical_card",
              "is_evolved", "source_confidence", "split", "label_source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metrics_for(true, predicted, confidences, classes):
    labels = list(range(len(classes)))
    report = classification_report(true, predicted, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    cannon = classes.index("cannon"); evolution = classes.index("cannon_evolution")
    confidence_array = np.asarray(confidences, dtype=np.float64)
    thresholds = {}
    for threshold in (.50, .70, .80, .90, .95):
        retained = np.flatnonzero(confidence_array >= threshold)
        correct = sum(predicted[i] == true[i] for i in retained)
        thresholds[f"{threshold:.2f}"] = {
            "retained": int(len(retained)), "rejected": int(len(true) - len(retained)),
            "coverage": float(len(retained) / len(true)),
            "accuracy": float(correct / len(retained)) if len(retained) else None,
            "total_correct": int(correct),
        }
    percentiles = {str(p): float(np.percentile(confidence_array, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    return {
        "sample_count": len(true), "total_correct": int(sum(a == b for a, b in zip(true, predicted))),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": {name: report[name] for name in classes},
        "cannon_accuracy": float(report["cannon"]["recall"]),
        "cannon_evolution_accuracy": float(report["cannon_evolution"]["recall"]),
        "cannon_samples": int(sum(value == cannon for value in true)),
        "cannon_evolution_samples": int(sum(value == evolution for value in true)),
        "confidence_percentiles": percentiles, "accuracy_at_confidence_threshold": thresholds,
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }


def plot_confusion(matrix, classes, path, title):
    matrix = np.asarray(matrix); figure, axis = plt.subplots(figsize=(11, 9)); image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right"); axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title(title)
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def contact_sheet(rows, path, title, limit):
    rows = rows[:limit]; columns, tile_width, tile_height, header = 6, 170, 215, 45
    sheet = np.full((header + max(1, int(np.ceil(len(rows)/columns))) * tile_height,
                     columns * tile_width, 3), 28, np.uint8)
    cv2.putText(sheet, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
    for index, row in enumerate(rows):
        image = cv2.imread(row["image_path"])
        if image is None: continue
        scale = min(148/image.shape[1], 150/image.shape[0]); image = cv2.resize(image, None, fx=scale, fy=scale)
        grid_row, column = divmod(index, columns); x = column*tile_width+(tile_width-image.shape[1])//2; y = header+grid_row*tile_height
        sheet[y:y+image.shape[0], x:x+image.shape[1]] = image
        cv2.putText(sheet, f"T:{row['visual_label']}"[:25], (column*tile_width+3,y+170), cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
        cv2.putText(sheet, f"P:{row['predicted_visual_label']} {float(row['confidence']):.2f}"[:27], (column*tile_width+3,y+190), cv2.FONT_HERSHEY_SIMPLEX,.34,(120,210,255),1)
    if not cv2.imwrite(str(path), sheet): raise RuntimeError(f"Could not write {path}")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.sheet_limit < 1: raise ValueError("Batch size and sheet limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, checkpoint = load_model_bundle(args.model, device)
    provenance = verify_provenance(args.matches, args.split_config, args.training_manifest, checkpoint)
    manifest_rows = []
    for match in args.matches:
        label_source = (args.dataset_root/match/"labels.csv")
        if not label_source.is_file(): label_source = args.classification_root/match/"classifications.csv"
        for row in read_match(match, args.dataset_root, args.classification_root):
            manifest_rows.append({**row, "split": "external", "label_source": str(label_source.resolve())})
    manifest_path = args.output / "external_manifest.csv"; write_manifest(manifest_rows, manifest_path)
    dataset = CardManifestDataset(manifest_path, "external", classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    true, predicted, confidences, indices = [], [], [], []
    with torch.inference_mode():
        for images, targets, sample_indices in loader:
            probabilities = model(images.to(device)).softmax(1).cpu(); confidence, prediction = probabilities.max(1)
            true.extend(targets.tolist()); predicted.extend(prediction.tolist()); confidences.extend(confidence.tolist()); indices.extend(sample_indices.tolist())
    prediction_rows = []
    for target, prediction, confidence, index in zip(true, predicted, confidences, indices):
        row = dataset.rows[index]; logical_card, is_evolved = logical_fields(classes[prediction])
        image = cv2.imread(row["image_path"]); saturation = None
        if image is not None:
            resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
            saturation = float(cv2.cvtColor(resized[18:101, 12:84], cv2.COLOR_BGR2HSV)[..., 1].mean())
        prediction_rows.append({**row, "predicted_visual_label": classes[prediction],
                                "predicted_logical_card": logical_card, "predicted_is_evolved": str(is_evolved).lower(),
                                "confidence": confidence, "correct": target == prediction,
                                "artwork_mean_saturation": saturation})
    with (args.output/"external_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    combined = metrics_for(true, predicted, confidences, classes); per_match = {}
    for match in args.matches:
        positions=[i for i,row in enumerate(prediction_rows) if row["match_id"]==match]
        per_match[match]=metrics_for([true[i] for i in positions],[predicted[i] for i in positions],[confidences[i] for i in positions],classes)
        plot_confusion(per_match[match]["confusion_matrix"],classes,args.output/f"confusion_matrix_{match}.png",f"External confusion matrix: {match}")
    electro_to_hog=[row for row in prediction_rows if row["visual_label"]=="electro_spirit" and row["predicted_visual_label"]=="hog_rider"]
    grey_repeat=[row for row in electro_to_hog if row["artwork_mean_saturation"] is not None and float(row["artwork_mean_saturation"])<28]
    payload={"checkpoint":str(args.model),"checkpoint_sha256":sha256(args.model),"device":str(device),
             "external_matches":args.matches,"provenance_verification":provenance,"classes":classes,
             "combined":combined,"per_match":per_match,
             "known_greyed_electro_spirit_error_check":{"electro_spirit_to_hog_rider":len(electro_to_hog),
                 "low_saturation_greyed_repeats":len(grey_repeat),"repeated":bool(grey_repeat)}}
    (args.output/"external_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    plot_confusion(combined["confusion_matrix"],classes,args.output/"confusion_matrix.png","Combined external confusion matrix")
    incorrect=sorted((row for row in prediction_rows if not row["correct"]),key=lambda row:float(row["confidence"]),reverse=True)
    low=sorted(prediction_rows,key=lambda row:float(row["confidence"]))
    contact_sheet(incorrect,args.output/"incorrect_predictions.jpg",f"External incorrect predictions - {len(incorrect)}",args.sheet_limit)
    contact_sheet(low,args.output/"low_confidence_predictions.jpg","Lowest-confidence external predictions",args.sheet_limit)
    print(json.dumps({"checkpoint_sha256":payload["checkpoint_sha256"],"combined":combined,
                      "per_match":{m:{k:v for k,v in metrics.items() if k in {"sample_count","total_correct","accuracy","macro_f1","cannon_accuracy","cannon_evolution_accuracy"}} for m,metrics in per_match.items()},
                      "known_error_check":payload["known_greyed_electro_spirit_error_check"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
