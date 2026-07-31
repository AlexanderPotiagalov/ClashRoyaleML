from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from card_reference_utils import (
        VISUAL_CLASSES,
        build_reference_banks,
        discover_slot_images,
        is_empty,
        load_normal_reference_paths,
        score_classes,
        timestamp_ms,
        visual_embedding,
        write_contact_sheet,
    )
except ModuleNotFoundError:  # Supports `python -m scripts...` and test imports.
    from scripts.card_reference_utils import (
    VISUAL_CLASSES,
    build_reference_banks,
    discover_slot_images,
    is_empty,
    load_normal_reference_paths,
    score_classes,
    timestamp_ms,
    visual_embedding,
    write_contact_sheet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify card slots with distinct normal and Evolution visual labels."
    )
    parser.add_argument("--slots-root", type=Path, default=Path("data/card_slots"))
    parser.add_argument(
        "--matches", nargs="+",
        default=["match_002", "match_003", "match_004", "match_005", "match_006"],
    )
    parser.add_argument(
        "--normal-labels", type=Path,
        default=Path("data/card_dataset/match_001/labels.csv"),
    )
    parser.add_argument(
        "--evolution-references", type=Path,
        default=Path("data/card_dataset/reference/cannon_evolution"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/card_classifications")
    )
    parser.add_argument("--min-similarity", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.025)
    parser.add_argument("--reference-limit", type=int, default=160)
    parser.add_argument("--contact-examples", type=int, default=36)
    return parser.parse_args()


def load_evolution_paths(directory: Path) -> list[Path]:
    manifest = directory / "approved_examples.csv"
    if not manifest.is_file():
        return []
    paths: list[Path] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "reference_path" not in reader.fieldnames:
            raise ValueError(f"Invalid Evolution reference manifest: {manifest}")
        for row in reader:
            path = Path(row["reference_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing approved Evolution reference: {path}")
            paths.append(path)
    return paths


def logical_fields(visual_label: str) -> tuple[str, bool]:
    if visual_label == "cannon_evolution":
        return "cannon", True
    if visual_label in {"empty", "unknown"}:
        return visual_label, False
    return visual_label, False


def main() -> int:
    args = parse_args()
    if "match_001" in args.matches:
        raise ValueError("match_001 is reference data and may not be an output target")
    if not 0 <= args.min_similarity <= 1 or not 0 <= args.min_margin <= 1:
        raise ValueError("Similarity thresholds must be in [0, 1]")
    if args.reference_limit < 1 or args.contact_examples < 1:
        raise ValueError("Example limits must be positive")

    grouped = load_normal_reference_paths(args.normal_labels)
    evolution_paths = load_evolution_paths(args.evolution_references)
    if evolution_paths:
        grouped["cannon_evolution"] = evolution_paths
    banks, selected = build_reference_banks(grouped, args.reference_limit)
    all_summaries: dict[str, object] = {}

    for match in args.matches:
        paths = discover_slot_images(args.slots_root / match)
        rows: list[dict[str, object]] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read: {path}")
            best_similarity = second_similarity = 0.0
            if is_empty(image):
                visual_label = "empty"
                confidence = 1.0
            else:
                scores = score_classes(visual_embedding(image), banks)
                best_similarity, best_label = scores[0]
                second_similarity = scores[1][0] if len(scores) > 1 else 0.0
                margin = best_similarity - second_similarity
                if best_similarity >= args.min_similarity and margin >= args.min_margin:
                    visual_label = best_label
                    confidence = float(np.clip(
                        (best_similarity - args.min_similarity) / max(1 - args.min_similarity, 1e-6)
                        + 2.0 * margin, 0, 1
                    ))
                else:
                    visual_label = "unknown"
                    confidence = 0.0
            logical_card, is_evolved = logical_fields(visual_label)
            rows.append({
                "image_path": str(path.resolve()), "match": match,
                "slot": path.parent.name, "filename": path.name,
                "timestamp_ms": timestamp_ms(path), "visual_label": visual_label,
                "logical_card": logical_card, "is_evolved": str(is_evolved).lower(),
                "confidence": confidence, "best_similarity": best_similarity,
                "second_similarity": second_similarity,
            })

        output = args.output_root / match
        output.mkdir(parents=True, exist_ok=True)
        fields = [
            "image_path", "match", "slot", "filename", "timestamp_ms",
            "visual_label", "logical_card", "is_evolved", "confidence",
            "best_similarity", "second_similarity",
        ]
        assignments = output / "classifications.csv"
        with assignments.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        counts = Counter(str(row["visual_label"]) for row in rows)
        sheets = {}
        for label in VISUAL_CLASSES:
            examples = [row for row in rows if row["visual_label"] == label]
            examples.sort(key=lambda row: (-float(row["confidence"]), int(row["timestamp_ms"])))
            display = []
            for row in examples:
                item = dict(row)
                item["display_label"] = f"{label} {float(row['confidence']):.2f}"
                item["display_detail"] = f"{row['slot']} {row['timestamp_ms']}ms"
                display.append(item)
            sheet = output / f"class_{label}.jpg"
            write_contact_sheet(display, sheet, f"{match} {label} - {len(examples)}", args.contact_examples)
            sheets[label] = str(sheet)
        summary = {
            "match": match, "image_count": len(rows),
            "visual_class_counts": {label: counts[label] for label in VISUAL_CLASSES},
            "normal_reference_counts": {
                label: len(paths) for label, paths in selected.items() if label != "cannon_evolution"
            },
            "cannon_evolution_reference_count": len(selected.get("cannon_evolution", [])),
            "classifications": str(assignments), "class_contact_sheets": sheets,
        }
        (output / "classification_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        all_summaries[match] = summary
        print(f"{match}: {len(rows)} images, {counts['cannon_evolution']} Cannon Evolution")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "classification_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    if not evolution_paths:
        print("WARNING: no approved Cannon Evolution references; that class remains inactive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
