from __future__ import annotations

import argparse
import csv
import itertools
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

try:
    from card_model_utils import VISUAL_CLASSES, logical_fields
except ModuleNotFoundError:
    from scripts.card_model_utils import VISUAL_CLASSES, logical_fields


EXCLUDED = {"unknown", "unknown_transition", "transition", "card_transition",
            "bad_crop", "partially_visible", "ambiguous"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a leakage-safe multi-match card training manifest.")
    parser.add_argument("--matches", nargs="+", default=[f"match_{i:03d}" for i in range(1, 7)])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/card_dataset"))
    parser.add_argument("--classification-root", type=Path, default=Path("data/card_classifications"))
    parser.add_argument("--output", type=Path, default=Path("data/card_training/card_training_manifest.csv"))
    parser.add_argument("--split-config", type=Path, default=Path("config/card_training_split.json"))
    parser.add_argument("--counts-output", type=Path, default=Path("data/card_training/class_counts.json"))
    parser.add_argument("--train-matches", nargs="*")
    parser.add_argument("--validation-match")
    parser.add_argument("--test-match")
    parser.add_argument("--fail-on-missing-eval-class", action="store_true")
    return parser.parse_args()


def read_match(match_id: str, dataset_root: Path, classification_root: Path) -> list[dict[str, object]]:
    dataset_csv = dataset_root / match_id / "labels.csv"
    classification_csv = classification_root / match_id / "classifications.csv"
    source = dataset_csv if dataset_csv.is_file() else classification_csv
    if not source.is_file():
        raise FileNotFoundError(f"No labels.csv or classifications.csv for {match_id}")
    with source.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        label = raw.get("visual_label", raw.get("label", "")).strip()
        if label in EXCLUDED or label not in VISUAL_CLASSES:
            continue
        image_path = Path(raw["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Labelled image is missing: {image_path}")
        logical_card, is_evolved = logical_fields(label)
        confidence = raw.get("confidence")
        if confidence in {None, ""}:
            confidence = "0.85" if raw.get("recovered", "false").lower() == "true" else "1.0"
        rows.append({
            "image_path": str(image_path.resolve()), "match_id": match_id,
            "slot": raw.get("slot", image_path.parent.name), "visual_label": label,
            "logical_card": logical_card, "is_evolved": str(is_evolved).lower(),
            "source_confidence": float(confidence),
        })
    return rows


def validate_split(split: dict[str, list[str]]) -> None:
    memberships = [match for matches in split.values() for match in matches]
    if len(memberships) != len(set(memberships)):
        raise ValueError("A match appears in more than one split")
    if not all(split.get(name) for name in ("train", "validation", "test")):
        raise ValueError("Train, validation, and test splits must all be non-empty")


def choose_split(matches: list[str], counts: dict[str, Counter]) -> dict[str, list[str]]:
    if len(matches) < 6:
        raise ValueError("At least six matches are required for a 4/1/1 split")
    best = None
    for validation, test in itertools.permutations(matches, 2):
        train = [match for match in matches if match not in {validation, test}]
        if len(train) != len(matches) - 2:
            continue
        train_counts = sum((counts[match] for match in train), Counter())
        if any(train_counts[label] == 0 for label in VISUAL_CLASSES):
            continue
        validation_coverage = sum(counts[validation][label] > 0 for label in VISUAL_CLASSES)
        test_coverage = sum(counts[test][label] > 0 for label in VISUAL_CLASSES)
        score = (
            min(validation_coverage, test_coverage), validation_coverage + test_coverage,
            min(counts[validation]["cannon_evolution"], counts[test]["cannon_evolution"]),
            counts[test]["cannon_evolution"], sum(counts[test].values()),
        )
        if best is None or score > best[0]:
            best = (score, {"train": train, "validation": [validation], "test": [test]})
    if best is None:
        raise ValueError("No match-level split places every class in training")
    return best[1]


def main() -> int:
    args = parse_args()
    rows_by_match = {match: read_match(match, args.dataset_root, args.classification_root) for match in args.matches}
    counts = {match: Counter(row["visual_label"] for row in rows) for match, rows in rows_by_match.items()}
    if args.train_matches or args.validation_match or args.test_match:
        if not (args.train_matches and args.validation_match and args.test_match):
            raise ValueError("Explicit split requires train, validation, and test arguments")
        split = {"train": args.train_matches, "validation": [args.validation_match], "test": [args.test_match]}
    else:
        split = choose_split(args.matches, counts)
    validate_split(split)
    split_for_match = {match: name for name, matches in split.items() for match in matches}
    missing_matches = set(args.matches) - set(split_for_match)
    if missing_matches:
        raise ValueError(f"Matches omitted from split: {sorted(missing_matches)}")
    split_counts = {}
    for split_name, split_matches in split.items():
        total = sum((counts[match] for match in split_matches), Counter())
        split_counts[split_name] = total
        missing = [label for label in VISUAL_CLASSES if total[label] == 0]
        if split_name == "train" and missing:
            raise ValueError(f"Training split lacks classes: {missing}")
        if split_name != "train" and missing:
            message = f"{split_name} split lacks important classes: {missing}"
            if args.fail_on_missing_eval_class:
                raise ValueError(message)
            warnings.warn(message)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "match_id", "slot", "visual_label", "logical_card",
              "is_evolved", "source_confidence", "split"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in args.matches:
            for row in rows_by_match[match]:
                writer.writerow({**row, "split": split_for_match[match]})
    split_payload = {
        "strategy": "complete_match_holdout", "visual_classes": VISUAL_CLASSES,
        "train_matches": split["train"], "validation_matches": split["validation"],
        "test_matches": split["test"],
    }
    args.split_config.parent.mkdir(parents=True, exist_ok=True)
    args.split_config.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    counts_payload = {
        "by_match": {m: {c: counts[m][c] for c in VISUAL_CLASSES} for m in args.matches},
        "by_split": {s: {c: split_counts[s][c] for c in VISUAL_CLASSES} for s in split_counts},
        "excluded_labels": sorted(EXCLUDED), "manifest_rows": sum(map(len, rows_by_match.values())),
    }
    args.counts_output.parent.mkdir(parents=True, exist_ok=True)
    args.counts_output.write_text(json.dumps(counts_payload, indent=2), encoding="utf-8")
    print(json.dumps(split_payload, indent=2))
    print(f"Manifest samples: {counts_payload['manifest_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
