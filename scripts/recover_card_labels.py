from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


TIMESTAMP_RE = re.compile(r"_t(\d+)ms(?:\.[^.]+)?$")
REQUIRED_FIELDS = {
    "image_path", "slot", "filename", "cluster_id", "unknown_reason"
}
BLOCKING_REASONS = {"transition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover unknown card labels when matching confident temporal anchors "
            "occur on both sides of an unknown run."
        )
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/card_clusters/match_001/assignments.csv"),
        help="Original clustering assignments (never modified)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("config/card_cluster_labels.json"),
        help="JSON mapping from cluster IDs to class labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/card_clusters/match_001"),
        help="Directory for recovered assignments and recovery reports",
    )
    parser.add_argument(
        "--max-gap-ms",
        type=int,
        default=2500,
        help="Maximum time from each unknown image to both matching anchors",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=36,
        help="Maximum examples in each recovery contact sheet",
    )
    return parser.parse_args()


def parse_timestamp(filename: str) -> int:
    match = TIMESTAMP_RE.search(filename)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {filename}")
    return int(match.group(1))


def load_labels(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label mapping: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Label mapping must be a non-empty JSON object")
    labels: dict[int, str] = {}
    for key, value in raw.items():
        try:
            cluster_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid cluster ID in mapping: {key!r}") from error
        if cluster_id < 0 or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid label mapping entry: {key!r}: {value!r}")
        labels[cluster_id] = value.strip()
    if len(set(labels.values())) != len(labels):
        raise ValueError("Every configured cluster must have a unique label")
    if "empty" not in labels.values():
        raise ValueError("Label mapping must include the empty class")
    return labels


def load_assignments(path: Path, labels: dict[int, str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing assignments file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Assignments CSV has no header: {path}")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Assignments CSV is missing fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError("Assignments CSV contains no rows")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        row["timestamp_ms"] = str(parse_timestamp(row["filename"]))
        key = (row["slot"], row["filename"])
        if key in seen:
            raise ValueError(f"Duplicate slot/filename assignment: {key}")
        seen.add(key)
        cluster_id = int(row["cluster_id"])
        if cluster_id >= 0 and cluster_id not in labels:
            raise ValueError(f"Cluster {cluster_id} has no configured label")
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Assigned image does not exist: {image_path}")
    return rows, fieldnames


def is_empty(row: dict[str, str], labels: dict[int, str]) -> bool:
    cluster_id = int(row["cluster_id"])
    return cluster_id >= 0 and labels[cluster_id] == "empty"


def is_transition(row: dict[str, str]) -> bool:
    return row["unknown_reason"].strip().lower() in BLOCKING_REASONS


def recover_rows(
    rows: list[dict[str, str]],
    labels: dict[int, str],
    max_gap_ms: int,
) -> list[dict[str, str]]:
    if max_gap_ms < 1:
        raise ValueError("--max-gap-ms must be positive")
    recovered = [dict(row) for row in rows]
    for row in recovered:
        row["original_cluster_id"] = row["cluster_id"]
        row["recovered"] = "False"
        row["recovery_reason"] = ""
        cluster_id = int(row["cluster_id"])
        row["label"] = labels[cluster_id] if cluster_id >= 0 else "unknown"

    slots: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        slots.setdefault(row["slot"], []).append(index)

    for indices in slots.values():
        indices.sort(key=lambda index: int(rows[index]["timestamp_ms"]))
        previous_anchor: list[int | None] = [None] * len(indices)
        next_anchor: list[int | None] = [None] * len(indices)

        anchor: int | None = None
        for position, index in enumerate(indices):
            row = rows[index]
            if is_empty(row, labels) or is_transition(row):
                anchor = None
            elif int(row["cluster_id"]) >= 0:
                anchor = index
            previous_anchor[position] = anchor

        anchor = None
        for position in range(len(indices) - 1, -1, -1):
            index = indices[position]
            row = rows[index]
            if is_empty(row, labels) or is_transition(row):
                anchor = None
            elif int(row["cluster_id"]) >= 0:
                anchor = index
            next_anchor[position] = anchor

        for position, index in enumerate(indices):
            row = rows[index]
            if int(row["cluster_id"]) >= 0 or is_transition(row):
                continue
            before = previous_anchor[position]
            after = next_anchor[position]
            if before is None or after is None or before == after:
                continue
            before_id = int(rows[before]["cluster_id"])
            after_id = int(rows[after]["cluster_id"])
            if before_id != after_id:
                continue
            timestamp = int(row["timestamp_ms"])
            before_gap = timestamp - int(rows[before]["timestamp_ms"])
            after_gap = int(rows[after]["timestamp_ms"]) - timestamp
            if before_gap < 0 or after_gap < 0:
                raise ValueError("Rows are not monotonic after timestamp sorting")
            if before_gap > max_gap_ms or after_gap > max_gap_ms:
                continue
            recovered[index]["cluster_id"] = str(before_id)
            recovered[index]["label"] = labels[before_id]
            recovered[index]["is_unknown"] = "False"
            recovered[index]["recovered"] = "True"
            recovered[index]["recovery_reason"] = (
                f"matching_temporal_anchors:{before_gap}ms/{after_gap}ms"
            )
    return recovered


def label_counts(rows: list[dict[str, str]], labels: dict[int, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        cluster_id = int(row["cluster_id"])
        counts[labels[cluster_id] if cluster_id >= 0 else "unknown"] += 1
    order = [*labels.values(), "unknown"]
    return {label: counts[label] for label in order}


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    limit: int,
) -> None:
    if limit < 1:
        raise ValueError("--examples must be positive")
    selected = rows[:limit]
    columns, tile_width, tile_height, header = 6, 160, 215, 48
    row_count = max(1, int(np.ceil(len(selected) / columns)))
    sheet = np.full(
        (header + row_count * tile_height, columns * tile_width, 3), 28, np.uint8
    )
    cv2.putText(
        sheet, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    for position, row in enumerate(selected):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(140 / w, 160 / h)
        display = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        grid_row, column = divmod(position, columns)
        x = column * tile_width + (tile_width - display.shape[1]) // 2
        y = header + grid_row * tile_height + 3
        sheet[y:y + display.shape[0], x:x + display.shape[1]] = display
        label = row.get("label", "unknown")
        detail = f"{row['slot']} {int(row['timestamp_ms'])}ms"
        cv2.putText(
            sheet, label[:22], (column * tile_width + 5, y + 178),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 230, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, detail, (column * tile_width + 5, y + 196),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 210, 210), 1, cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def main() -> int:
    args = parse_args()
    if args.max_gap_ms < 1:
        raise ValueError("--max-gap-ms must be positive")
    if args.examples < 1:
        raise ValueError("--examples must be positive")
    labels = load_labels(args.labels)
    rows, original_fields = load_assignments(args.assignments, labels)
    recovered = recover_rows(rows, labels, args.max_gap_ms)

    output_fields = [
        *original_fields,
        "timestamp_ms", "original_cluster_id", "label", "recovered", "recovery_reason",
    ]
    output_fields = list(dict.fromkeys(output_fields))
    recovered_path = args.output_dir / "recovered_assignments.csv"
    write_csv(recovered_path, recovered, output_fields)

    unresolved = [row for row in recovered if int(row["cluster_id"]) < 0]
    recovered_examples = [row for row in recovered if row["recovered"] == "True"]
    write_csv(args.output_dir / "remaining_unknowns.csv", unresolved, output_fields)
    contact_sheet(
        recovered_examples, args.output_dir / "recovered_examples.jpg",
        f"Safely recovered - {len(recovered_examples)} images", args.examples,
    )
    contact_sheet(
        unresolved, args.output_dir / "unresolved_examples.jpg",
        f"Still unknown - {len(unresolved)} images", args.examples,
    )

    before = label_counts(rows, labels)
    after = label_counts(recovered, labels)
    recovered_by_label = Counter(row["label"] for row in recovered_examples)
    summary = {
        "source_assignments": str(args.assignments),
        "label_mapping": str(args.labels),
        "max_gap_ms": args.max_gap_ms,
        "image_count": len(rows),
        "recovered_count": len(recovered_examples),
        "remaining_unknown_count": len(unresolved),
        "counts_before": before,
        "counts_after": after,
        "recovered_by_label": dict(sorted(recovered_by_label.items())),
        "outputs": {
            "recovered_assignments": str(recovered_path),
            "remaining_unknowns": str(args.output_dir / "remaining_unknowns.csv"),
            "recovered_contact_sheet": str(args.output_dir / "recovered_examples.jpg"),
            "unresolved_contact_sheet": str(args.output_dir / "unresolved_examples.jpg"),
        },
    }
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Images:             {len(rows)}")
    print(f"Unknown before:     {before['unknown']}")
    print(f"Safely recovered:   {len(recovered_examples)}")
    print(f"Unknown remaining:  {len(unresolved)}")
    print(f"Recovered output:   {recovered_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
