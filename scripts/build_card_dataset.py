from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from recover_card_labels import load_labels


REQUIRED_FIELDS = {
    "image_path", "slot", "filename", "cluster_id", "label",
    "timestamp_ms", "recovered",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a labeled card-image dataset from recovered assignments, using "
            "hard links when supported and copies as a safe fallback."
        )
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/card_clusters/match_001/recovered_assignments.csv"),
        help="Recovered assignments produced by recover_card_labels.py",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("config/card_cluster_labels.json"),
        help="JSON cluster-to-label mapping",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/card_dataset/match_001"),
        help="Dataset output directory",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Always copy images instead of attempting hard links",
    )
    return parser.parse_args()


def load_rows(path: Path, labels: dict[int, str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing recovered assignments: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Recovered assignments has no header: {path}")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Recovered assignments is missing fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Recovered assignments contains no rows")
    for row in rows:
        cluster_id = int(row["cluster_id"])
        expected = labels.get(cluster_id, "unknown")
        if row["label"] != expected:
            raise ValueError(
                f"Label mismatch for {row['slot']}/{row['filename']}: "
                f"{row['label']!r} != {expected!r}"
            )
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Missing source image: {row['image_path']}")
    return rows


def materialize(source: Path, destination: Path, copy_only: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Existing dataset file differs from source: {destination}"
            )
        try:
            if os.path.samefile(source, destination):
                return "existing_hardlink"
        except OSError:
            pass
        return "existing_file"
    if not copy_only:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    rows = load_rows(args.assignments, labels)
    args.output.mkdir(parents=True, exist_ok=True)

    class_names = [*labels.values(), "unknown"]
    for class_name in class_names:
        (args.output / class_name).mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, str]] = []
    methods: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    destinations: set[Path] = set()
    for row in rows:
        label = row["label"]
        source = Path(row["image_path"]).resolve()
        destination = args.output / label / f"{row['slot']}__{row['filename']}"
        if destination in destinations:
            raise ValueError(f"Duplicate dataset destination: {destination}")
        destinations.add(destination)
        method = materialize(source, destination, args.copy_only)
        methods[method] += 1
        class_counts[label] += 1
        output_rows.append(
            {
                "dataset_path": str(destination.resolve()),
                "image_path": str(source),
                "label": label,
                "cluster_id": row["cluster_id"],
                "slot": row["slot"],
                "filename": row["filename"],
                "timestamp_ms": row["timestamp_ms"],
                "recovered": row["recovered"],
                "link_method": method,
            }
        )

    labels_path = args.output / "labels.csv"
    fields = [
        "dataset_path", "image_path", "label", "cluster_id", "slot",
        "filename", "timestamp_ms", "recovered", "link_method",
    ]
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "source_assignments": str(args.assignments),
        "label_mapping": str(args.labels),
        "output_directory": str(args.output),
        "image_count": len(output_rows),
        "class_count": len(class_names),
        "class_counts": {name: class_counts[name] for name in class_names},
        "materialization": dict(sorted(methods.items())),
        "labels_csv": str(labels_path),
    }
    (args.output / "class_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Dataset images:  {len(output_rows)}")
    print(f"Classes:         {len(class_names)}")
    print(f"Labels CSV:      {labels_path}")
    print(f"Materialization: {dict(methods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
