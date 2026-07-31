from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = (
    "empty",
    "card_transition",
    "greyed_or_unaffordable",
    "partially_visible",
    "animation_or_glow",
    "bad_crop",
    "visually_stable_unknown",
)
REQUIRED_UNKNOWN_FIELDS = {
    "image_path", "slot", "filename", "timestamp_ms", "cluster_id", "unknown_reason"
}
REQUIRED_CONTEXT_FIELDS = REQUIRED_UNKNOWN_FIELDS | {"label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize unresolved card-slot images using visual statistics, "
            "nearest labelled frames, temporal context, and class prototypes."
        )
    )
    parser.add_argument(
        "--unknowns",
        type=Path,
        default=Path("data/card_clusters/match_001/remaining_unknowns.csv"),
        help="Unresolved rows produced by recover_card_labels.py",
    )
    parser.add_argument(
        "--context-assignments",
        type=Path,
        default=Path("data/card_clusters/match_001/recovered_assignments.csv"),
        help="Assignments containing labelled temporal context and prototype images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/card_clusters/match_001/unknown_analysis"),
        help="Separate output directory; source assignments are never modified",
    )
    parser.add_argument(
        "--temporal-gap-ms",
        type=int,
        default=4000,
        help="Maximum distance used when interpreting neighbouring labelled frames",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=36,
        help="Maximum images per category and class contact sheet",
    )
    parser.add_argument(
        "--prototype-examples",
        type=int,
        default=160,
        help="Maximum confident labelled images used to form each class prototype",
    )
    return parser.parse_args()


def load_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    for row in rows:
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Missing image: {row['image_path']}")
        int(row["timestamp_ms"])
        int(row["cluster_id"])
    return rows, fields


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def unit_norm(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def image_metrics(image: np.ndarray) -> dict[str, object]:
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    center = hsv[24:105, 15:81]
    blue = (
        (center[:, :, 0] >= 90)
        & (center[:, :, 0] <= 125)
        & (center[:, :, 1] >= 80)
    )
    side_edges = np.concatenate(
        [edges[18:120, 2:15].ravel(), edges[18:120, 81:94].ravel()]
    ).mean() / 255.0
    artwork = resized[18:101, 12:84].astype(np.float32)
    colorfulness = float(np.ptp(artwork, axis=2).mean())
    feature_image = cv2.resize(resized[14:108, 8:88], (24, 30), cv2.INTER_AREA)
    lab = cv2.cvtColor(feature_image, cv2.COLOR_BGR2LAB)
    feature_gray = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
    feature_edges = cv2.Canny(feature_gray, 50, 130)
    lab_feature = (
        lab.astype(np.float32) - np.array([128.0, 128.0, 128.0], np.float32)
    ) / 128.0
    feature = unit_norm(
        np.concatenate([2.0 * lab_feature.ravel(), feature_edges.astype(np.float32).ravel() / 255.0])
    )
    return {
        "resized": resized,
        "gray_feature": normalized(cv2.resize(gray, (24, 32), cv2.INTER_AREA)),
        "edge_feature": normalized(cv2.resize(edges, (24, 32), cv2.INTER_AREA)),
        "prototype_feature": feature,
        "brightness": float(hsv[:, :, 2].mean()),
        "saturation": float(hsv[:, :, 1].mean()),
        "colorfulness": colorfulness,
        "edge_density": float((edges > 0).mean()),
        "card_edge_score": float(side_edges),
        "blue_fraction": float(blue.mean()),
        "texture": float(cv2.Laplacian(center[:, :, 2], cv2.CV_32F).var()),
    }


def similarity(left: dict[str, object], right: dict[str, object]) -> tuple[float, float]:
    pixel = float(np.dot(left["gray_feature"], right["gray_feature"]))
    edge = float(np.dot(left["edge_feature"], right["edge_feature"]))
    return pixel, edge


def make_context_index(
    rows: list[dict[str, str]],
) -> dict[str, tuple[list[int], list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["cluster_id"]) >= 0:
            grouped.setdefault(row["slot"], []).append(row)
    result = {}
    for slot, slot_rows in grouped.items():
        slot_rows.sort(key=lambda row: int(row["timestamp_ms"]))
        result[slot] = ([int(row["timestamp_ms"]) for row in slot_rows], slot_rows)
    return result


def nearest_context(
    row: dict[str, str],
    context: dict[str, tuple[list[int], list[dict[str, str]]]],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    times, rows = context.get(row["slot"], ([], []))
    timestamp = int(row["timestamp_ms"])
    position = bisect.bisect_left(times, timestamp)
    before = rows[position - 1] if position else None
    after = rows[position] if position < len(rows) else None
    return before, after


def build_prototypes(
    rows: list[dict[str, str]],
    metrics: dict[str, dict[str, object]],
    limit: int,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        if int(row["cluster_id"]) < 0 or row["label"] in {"empty", "unknown"}:
            continue
        # Original confident labels are preferable to temporally recovered examples.
        if row.get("original_cluster_id", row["cluster_id"]) == "-1":
            continue
        values = grouped.setdefault(row["label"], [])
        if len(values) < limit:
            values.append(metrics[row["image_path"]]["prototype_feature"])
    prototypes: dict[str, np.ndarray] = {
        label: np.vstack(features) for label, features in grouped.items()
    }
    if not prototypes:
        raise ValueError("No confident labelled rows were available for prototypes")
    return prototypes


def prototype_suggestion(
    metric: dict[str, object], prototypes: dict[str, np.ndarray]
) -> tuple[str, float, float]:
    feature = metric["prototype_feature"]
    scored: list[tuple[float, str]] = []
    for label, prototype_bank in prototypes.items():
        similarities = np.sort(prototype_bank @ feature)[::-1]
        scored.append((float(similarities[: min(3, len(similarities))].mean()), label))
    scored.sort(reverse=True)
    best_score, best_label = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    confidence = float(np.clip((best_score - second_score) * 2.5 + (best_score - 0.45), 0, 1))
    return best_label, confidence, best_score


def categorize(
    row: dict[str, str],
    metric: dict[str, object],
    before: dict[str, str] | None,
    after: dict[str, str] | None,
    metric_cache: dict[str, dict[str, object]],
    adjacent_stable: bool,
    temporal_gap_ms: int,
) -> tuple[str, str]:
    timestamp = int(row["timestamp_ms"])
    before_gap = timestamp - int(before["timestamp_ms"]) if before else 10**12
    after_gap = int(after["timestamp_ms"]) - timestamp if after else 10**12
    before_close = before is not None and 0 <= before_gap <= temporal_gap_ms
    after_close = after is not None and 0 <= after_gap <= temporal_gap_ms
    before_sim = before_edge = after_sim = after_edge = -1.0
    if before:
        before_sim, before_edge = similarity(metric, metric_cache[before["image_path"]])
    if after:
        after_sim, after_edge = similarity(metric, metric_cache[after["image_path"]])
    max_pixel, max_edge = max(before_sim, after_sim), max(before_edge, after_edge)
    labels_differ = before_close and after_close and before["label"] != after["label"]
    same_label = before_close and after_close and before["label"] == after["label"]

    if metric["blue_fraction"] > 0.72 and metric["texture"] < 220:
        return "empty", "blue_placeholder"
    if row["unknown_reason"].strip().lower() == "transition":
        return "card_transition", "detected_transition"
    if labels_differ:
        return "card_transition", "different_nearby_labels"
    if metric["card_edge_score"] < 0.07 and (metric["edge_density"] < 0.06 or max_edge < 0.15):
        return "bad_crop", "weak_card_geometry"
    if metric["colorfulness"] < 15 and max_edge > 0.25:
        return "greyed_or_unaffordable", "low_colour_with_matching_edges"
    if (
        metric["brightness"] > 185 or metric["saturation"] > 150
    ) and max_edge > 0.2:
        return "animation_or_glow", "extreme_colour_or_brightness"
    if same_label and (max_pixel < 0.76 or max_edge < 0.28):
        return "partially_visible", "matching_context_but_incomplete_visual"
    if adjacent_stable or (same_label and max_pixel >= 0.76 and max_edge >= 0.28):
        return "visually_stable_unknown", "stable_visual_sequence"
    if metric["card_edge_score"] >= 0.10 and max_edge >= 0.18:
        return "partially_visible", "card_structure_with_weak_match"
    return "bad_crop", "no_reliable_card_structure"


def write_contact_sheet(
    rows: list[dict[str, str]], output: Path, title: str, limit: int
) -> None:
    selected = rows[:limit]
    columns, tile_width, tile_height, header = 6, 160, 220, 48
    row_count = max(1, int(np.ceil(len(selected) / columns)))
    sheet = np.full((header + row_count * tile_height, columns * tile_width, 3), 28, np.uint8)
    cv2.putText(sheet, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2, cv2.LINE_AA)
    for position, row in enumerate(selected):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(140 / w, 160 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), cv2.INTER_AREA)
        grid_row, column = divmod(position, columns)
        x = column * tile_width + (tile_width - image.shape[1]) // 2
        y = header + grid_row * tile_height + 3
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        primary = row.get("suggested_label") or row.get("label") or "unknown"
        confidence = row.get("suggestion_confidence", "")
        if confidence:
            primary += f" {float(confidence):.2f}"
        cv2.putText(sheet, primary[:22], (column * tile_width + 4, y + 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 230, 255), 1, cv2.LINE_AA)
        detail = f"{row['slot']} {row['timestamp_ms']}ms"
        cv2.putText(sheet, detail, (column * tile_width + 4, y + 197),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (210, 210, 210), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output}")


def main() -> int:
    args = parse_args()
    if args.temporal_gap_ms < 1 or args.examples < 1 or args.prototype_examples < 1:
        raise ValueError("Gap and example counts must all be positive")
    unknowns, unknown_fields = load_csv(args.unknowns, REQUIRED_UNKNOWN_FIELDS)
    context_rows, _ = load_csv(args.context_assignments, REQUIRED_CONTEXT_FIELDS)
    if any(int(row["cluster_id"]) >= 0 for row in unknowns):
        raise ValueError("remaining_unknowns.csv contains a labelled row")

    all_paths = {row["image_path"] for row in [*unknowns, *context_rows]}
    metric_cache: dict[str, dict[str, object]] = {}
    for path in sorted(all_paths):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        metric_cache[path] = image_metrics(image)

    context = make_context_index(context_rows)
    prototypes = build_prototypes(context_rows, metric_cache, args.prototype_examples)
    unknown_by_slot: dict[str, list[dict[str, str]]] = {}
    for row in unknowns:
        unknown_by_slot.setdefault(row["slot"], []).append(row)
    stable_keys: set[tuple[str, str]] = set()
    for slot_rows in unknown_by_slot.values():
        slot_rows.sort(key=lambda row: int(row["timestamp_ms"]))
        for left, right in zip(slot_rows, slot_rows[1:]):
            gap = int(right["timestamp_ms"]) - int(left["timestamp_ms"])
            pixel, edge = similarity(metric_cache[left["image_path"]], metric_cache[right["image_path"]])
            if gap <= 1100 and pixel >= 0.88 and edge >= 0.55:
                stable_keys.add((left["slot"], left["filename"]))
                stable_keys.add((right["slot"], right["filename"]))

    analyzed: list[dict[str, str]] = []
    for source_row in unknowns:
        row = dict(source_row)
        before, after = nearest_context(row, context)
        metric = metric_cache[row["image_path"]]
        category, reason = categorize(
            row, metric, before, after, metric_cache,
            (row["slot"], row["filename"]) in stable_keys,
            args.temporal_gap_ms,
        )
        row["visual_category"] = category
        row["category_reason"] = reason
        row["nearest_before_label"] = before["label"] if before else ""
        row["nearest_before_gap_ms"] = (
            str(int(row["timestamp_ms"]) - int(before["timestamp_ms"])) if before else ""
        )
        row["nearest_after_label"] = after["label"] if after else ""
        row["nearest_after_gap_ms"] = (
            str(int(after["timestamp_ms"]) - int(row["timestamp_ms"])) if after else ""
        )
        row["brightness"] = f"{metric['brightness']:.4f}"
        row["saturation"] = f"{metric['saturation']:.4f}"
        row["colorfulness"] = f"{metric['colorfulness']:.4f}"
        row["edge_density"] = f"{metric['edge_density']:.6f}"
        row["card_edge_score"] = f"{metric['card_edge_score']:.6f}"
        row["suggested_label"] = ""
        row["suggestion_confidence"] = ""
        row["prototype_similarity"] = ""
        if category == "visually_stable_unknown":
            label, confidence, score = prototype_suggestion(metric, prototypes)
            row["suggested_label"] = label
            row["suggestion_confidence"] = f"{confidence:.6f}"
            row["prototype_similarity"] = f"{score:.6f}"
        analyzed.append(row)

    output_fields = list(dict.fromkeys([
        *unknown_fields, "visual_category", "category_reason",
        "nearest_before_label", "nearest_before_gap_ms",
        "nearest_after_label", "nearest_after_gap_ms", "brightness", "saturation",
        "colorfulness", "edge_density", "card_edge_score", "suggested_label",
        "suggestion_confidence", "prototype_similarity",
    ]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv = args.output_dir / "unknown_analysis.csv"
    with analysis_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analyzed)

    category_counts = Counter(row["visual_category"] for row in analyzed)
    category_sheets: dict[str, str] = {}
    for category in CATEGORIES:
        category_rows = [row for row in analyzed if row["visual_category"] == category]
        sheet = args.output_dir / f"unknown_{category}.jpg"
        write_contact_sheet(category_rows, sheet, f"{category} - {len(category_rows)}", args.examples)
        category_sheets[category] = str(sheet)

    class_counts: Counter[str] = Counter()
    class_sheets: dict[str, str] = {}
    labels = sorted({row["label"] for row in context_rows if int(row["cluster_id"]) >= 0})
    for label in labels:
        label_rows = [
            row for row in context_rows
            if int(row["cluster_id"]) >= 0 and row["label"] == label
        ]
        label_rows.sort(key=lambda row: (row["slot"], int(row["timestamp_ms"])))
        if len(label_rows) > args.examples:
            positions = np.linspace(0, len(label_rows) - 1, args.examples).astype(int)
            examples = [label_rows[position] for position in positions]
        else:
            examples = label_rows
        sheet = args.output_dir / f"class_{label}.jpg"
        write_contact_sheet(examples, sheet, f"class {label} - {len(label_rows)}", args.examples)
        class_counts[label] = len(label_rows)
        class_sheets[label] = str(sheet)

    suggestions = [row for row in analyzed if row["suggested_label"]]
    suggestion_counts = Counter(row["suggested_label"] for row in suggestions)
    summary = {
        "source_unknowns": str(args.unknowns),
        "context_assignments": str(args.context_assignments),
        "unknown_count": len(analyzed),
        "category_counts": {category: category_counts[category] for category in CATEGORIES},
        "stable_suggestion_count": len(suggestions),
        "suggestions_by_label": dict(sorted(suggestion_counts.items())),
        "prototype_labels": sorted(prototypes),
        "class_counts": dict(sorted(class_counts.items())),
        "parameters": {
            "temporal_gap_ms": args.temporal_gap_ms,
            "examples": args.examples,
            "prototype_examples": args.prototype_examples,
        },
        "outputs": {
            "analysis_csv": str(analysis_csv),
            "category_contact_sheets": category_sheets,
            "class_contact_sheets": class_sheets,
        },
        "identity_assignment_policy": (
            "Suggestions are review-only; all analyzed rows remain cluster_id=-1 and label=unknown."
        ),
    }
    (args.output_dir / "unknown_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Unknown images analyzed: {len(analyzed)}")
    print(f"Categories:              {dict(category_counts)}")
    print(f"Stable suggestions:      {len(suggestions)} (review only)")
    print(f"Output directory:        {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
