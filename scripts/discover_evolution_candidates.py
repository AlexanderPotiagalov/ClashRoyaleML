from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from card_reference_utils import (
        bank_similarity,
        build_reference_banks,
        discover_slot_images,
        load_normal_reference_paths,
        score_classes,
        timestamp_ms,
        visual_embedding,
        visual_stats,
        write_contact_sheet,
    )
except ModuleNotFoundError:  # Supports `python -m scripts...` and test imports.
    from scripts.card_reference_utils import (
    bank_similarity,
    build_reference_banks,
    discover_slot_images,
    load_normal_reference_paths,
    score_classes,
    timestamp_ms,
    visual_embedding,
    visual_stats,
    write_contact_sheet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find review-only Cannon Evolution candidates in new matches by comparing "
            "them with normal match_001 card references."
        )
    )
    parser.add_argument(
        "--slots-root", type=Path, default=Path("data/card_slots"),
        help="Root containing match_002, match_003, etc.",
    )
    parser.add_argument(
        "--matches", nargs="+",
        default=["match_002", "match_003", "match_004", "match_005", "match_006"],
        help="Match directory names to search",
    )
    parser.add_argument(
        "--reference-labels", type=Path,
        default=Path("data/card_dataset/match_001/labels.csv"),
        help="Existing match_001 labelled dataset CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/card_clusters/evolution_discovery"),
        help="Output directory for candidates and contact sheet",
    )
    parser.add_argument(
        "--review-file", type=Path,
        default=Path("config/cannon_evolution_review.csv"),
        help="Manual review CSV created or updated without losing statuses",
    )
    parser.add_argument(
        "--group-review-file", type=Path,
        default=Path("config/cannon_evolution_group_review.csv"),
        help="Condensed review CSV with one row per visually distinct group",
    )
    parser.add_argument("--min-cannon-similarity", type=float, default=0.62)
    parser.add_argument("--min-confidence", type=float, default=0.42)
    parser.add_argument("--reference-limit", type=int, default=160)
    parser.add_argument("--contact-examples", type=int, default=72)
    return parser.parse_args()


def candidate_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_existing_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "status", "notes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid review CSV: {path}")
        reviews = {row["candidate_id"]: dict(row) for row in reader}
    for review in reviews.values():
        if review["status"] not in {"approved", "rejected", "uncertain"}:
            raise ValueError(f"Invalid review status: {review['status']!r}")
    return reviews


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_cannon_similarity <= 1 or not 0 <= args.min_confidence <= 1:
        raise ValueError("Similarity and confidence thresholds must be in [0, 1]")
    if args.reference_limit < 1 or args.contact_examples < 1:
        raise ValueError("Reference and contact example counts must be positive")

    grouped = load_normal_reference_paths(args.reference_labels)
    banks, selected = build_reference_banks(grouped, args.reference_limit)
    cannon_bank = banks["cannon"]
    cannon_stats = []
    for path in selected["cannon"]:
        image = cv2.imread(str(path))
        if image is not None:
            cannon_stats.append(visual_stats(image))
    normal_stats = {
        key: float(np.median([row[key] for row in cannon_stats])) for key in cannon_stats[0]
    }

    records: list[dict[str, object]] = []
    feature_by_path: dict[Path, np.ndarray] = {}
    paths_by_slot: dict[tuple[str, str], list[Path]] = {}
    for match in args.matches:
        paths = discover_slot_images(args.slots_root / match)
        for path in paths:
            paths_by_slot.setdefault((match, path.parent.name), []).append(path)
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read: {path}")
            feature_by_path[path] = visual_embedding(image)

    for paths in paths_by_slot.values():
        paths.sort(key=timestamp_ms)
    cannon_score_by_path = {
        path: bank_similarity(feature, cannon_bank)
        for path, feature in feature_by_path.items()
    }
    neighbour_stability: dict[Path, float] = {}
    for paths in paths_by_slot.values():
        for position, path in enumerate(paths):
            values = []
            for offset in (-1, 1):
                other_position = position + offset
                if 0 <= other_position < len(paths):
                    values.append(float(feature_by_path[path] @ feature_by_path[paths[other_position]]))
            neighbour_stability[path] = float(np.mean(values)) if values else 0.0

    for (match, slot), paths in paths_by_slot.items():
        for position, path in enumerate(paths):
            feature = feature_by_path[path]
            scores = score_classes(feature, banks)
            by_label = {label: score for score, label in scores}
            cannon_similarity = by_label["cannon"]
            cannon_rank = next(index for index, (_, label) in enumerate(scores, 1) if label == "cannon")
            strongest_other = max(score for score, label in scores if label != "cannon")
            image = cv2.imread(str(path))
            stats = visual_stats(image)
            resized = cv2.resize(image, (96, 128), cv2.INTER_AREA)
            art_hsv = cv2.cvtColor(resized[18:101, 12:84], cv2.COLOR_BGR2HSV)
            art_saturation = float(art_hsv[..., 1].mean())
            cyan_fraction = float(((art_hsv[..., 0] >= 75) & (art_hsv[..., 0] <= 105)
                                   & (art_hsv[..., 1] > 85) & (art_hsv[..., 2] > 70)).mean())
            neon_purple_fraction = float(((art_hsv[..., 0] >= 125) & (art_hsv[..., 0] <= 175)
                                          & (art_hsv[..., 1] > 95) & (art_hsv[..., 2] > 110)).mean())
            temporal_cannon_similarity = max(
                cannon_score_by_path[paths[index]]
                for index in range(max(0, position - 20), min(len(paths), position + 21))
            )
            edges = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 60, 140)
            card_edge_score = float(np.concatenate([
                edges[18:120, 2:15].ravel(), edges[18:120, 81:94].ravel()
            ]).mean() / 255.0)
            outlier_score = float(np.clip((0.94 - cannon_similarity) / 0.45, 0, 1))
            purple_delta = max(0.0, stats["purple_border_fraction"] - normal_stats["purple_border_fraction"])
            evolution_appearance = float(np.clip(
                2.8 * cyan_fraction + 3.5 * neon_purple_fraction + 4.0 * purple_delta,
                0, 1,
            ))
            saturation_delta = max(0.0, stats["saturation"] - normal_stats["saturation"]) / 90.0
            banner_delta = abs(stats["orange_banner_fraction"] - normal_stats["orange_banner_fraction"])
            glow_score = float(np.clip(8.0 * purple_delta + saturation_delta + 2.5 * banner_delta, 0, 1))
            marker_score = float(np.clip(
                12.0 * max(0.0, stats["orange_banner_fraction"] - normal_stats["orange_banner_fraction"])
                + 15.0 * purple_delta, 0, 1
            ))
            temporal_score = float(np.clip((neighbour_stability[path] - 0.72) / 0.25, 0, 1))
            cannon_likeness = float(np.clip((cannon_similarity - strongest_other + 0.12) / 0.24, 0, 1))
            confidence = float(np.clip(
                0.34 * outlier_score + 0.29 * glow_score
                + 0.20 * temporal_score + 0.17 * cannon_likeness,
                0, 1,
            ))
            confidence = max(confidence, 0.55 * marker_score)
            normal_outlier_candidate = (
                cannon_similarity >= args.min_cannon_similarity
                and cannon_rank == 1
                and card_edge_score >= 0.12
                and (
                    (confidence >= args.min_confidence and (outlier_score >= 0.10 or glow_score >= 0.12))
                    or marker_score >= 0.35
                )
            )
            evolution_colour_candidate = (
                card_edge_score >= 0.12
                and evolution_appearance >= 0.34
                and temporal_cannon_similarity >= 0.62
                and (cyan_fraction >= 0.045 or neon_purple_fraction >= 0.035)
            )
            if normal_outlier_candidate or evolution_colour_candidate:
                records.append({
                    "candidate_id": candidate_id(path),
                    "image_path": str(path.resolve()),
                    "match": match,
                    "slot": slot,
                    "timestamp_ms": timestamp_ms(path),
                    "normal_cannon_similarity": cannon_similarity,
                    "nearest_normal_label": scores[0][1],
                    "nearest_normal_similarity": scores[0][0],
                    "cannon_rank": cannon_rank,
                    "outlier_score": outlier_score,
                    "glow_difference": glow_score,
                    "evolution_marker_score": marker_score,
                    "temporal_stability": neighbour_stability[path],
                    "suggested_confidence": confidence,
                    "brightness": stats["brightness"],
                    "saturation": stats["saturation"],
                    "art_saturation": art_saturation,
                    "cyan_fraction": cyan_fraction,
                    "neon_purple_fraction": neon_purple_fraction,
                    "evolution_appearance": evolution_appearance,
                    "temporal_cannon_similarity": temporal_cannon_similarity,
                })

    records.sort(key=lambda row: (-float(row["suggested_confidence"]), row["match"], int(row["timestamp_ms"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id", "image_path", "match", "slot", "timestamp_ms",
        "normal_cannon_similarity", "nearest_normal_label", "nearest_normal_similarity",
        "cannon_rank", "outlier_score", "glow_difference", "evolution_marker_score", "temporal_stability",
        "suggested_confidence", "brightness", "saturation", "art_saturation",
        "cyan_fraction", "neon_purple_fraction", "evolution_appearance",
        "temporal_cannon_similarity",
    ]
    candidates_path = args.output_dir / "evolution_candidates.csv"
    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    display_rows = []
    for row in records:
        display = dict(row)
        display["display_label"] = f"conf={float(row['suggested_confidence']):.2f} sim={float(row['normal_cannon_similarity']):.2f}"
        display["display_detail"] = f"{row['match']} {row['slot']} {row['timestamp_ms']}ms"
        display_rows.append(display)
    write_contact_sheet(
        display_rows, args.output_dir / "cannon_evolution_candidates.jpg",
        f"Cannon Evolution candidates - {len(records)} review required", args.contact_examples,
    )

    # Collapse near-identical frames. Artwork embeddings deliberately ignore most
    # cycle-marker chrome, so repeated frames from a temporal run land together.
    resolved_features = {path.resolve(): feature for path, feature in feature_by_path.items()}
    groups: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    for row in records:
        feature = resolved_features[Path(str(row["image_path"]))]
        best_index = -1
        best_similarity = -1.0
        for index, group in enumerate(groups):
            similarity = float(feature @ group["centroid"])
            if similarity > best_similarity:
                best_index, best_similarity = index, similarity
        if best_similarity < 0.970:
            groups.append({"centroid": feature.copy(), "rows": [row]})
            best_index = len(groups) - 1
        else:
            group_rows = groups[best_index]["rows"]
            group_rows.append(row)
            centroid = np.mean([resolved_features[Path(str(item["image_path"]))] for item in group_rows], axis=0)
            groups[best_index]["centroid"] = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        membership.append({"candidate_id": row["candidate_id"], "group_index": best_index})

    group_records = []
    group_id_by_index = {}
    for index, group in enumerate(groups):
        group_rows = group["rows"]
        representative = max(group_rows, key=lambda item: float(item["suggested_confidence"]))
        group_id = f"evo_group_{index + 1:03d}"
        group_id_by_index[index] = group_id
        greyed = float(np.median([float(item["art_saturation"]) for item in group_rows])) < 28.0
        lacks_evo_colours = float(np.median([
            float(item["evolution_appearance"]) for item in group_rows
        ])) < 0.34
        auto_rejected = greyed or lacks_evo_colours
        group_records.append({
            "group_id": group_id,
            "representative_path": representative["image_path"],
            "image_count": len(group_rows),
            "matches": ";".join(sorted({str(item["match"]) for item in group_rows})),
            "suggested_confidence": representative["suggested_confidence"],
            "status": "rejected" if auto_rejected else "uncertain",
            "notes": ("auto-filtered: greyed/unaffordable" if greyed else
                      "auto-filtered: lacks Evo cyan/purple artwork" if lacks_evo_colours else ""),
        })
    membership_path = args.output_dir / "evolution_candidate_group_members.csv"
    member_by_id = {str(item["candidate_id"]): item for item in records}
    member_fields = ["candidate_id", "image_path", "match", "slot", "timestamp_ms"]
    with membership_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["group_id", *member_fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in membership:
            row = member_by_id[str(item["candidate_id"])]
            writer.writerow({"group_id": group_id_by_index[int(item["group_index"])], **{key: row[key] for key in member_fields}})

    old_groups = load_existing_reviews(args.group_review_file) if args.group_review_file.exists() else {}
    args.group_review_file.parent.mkdir(parents=True, exist_ok=True)
    group_fields = ["candidate_id", "representative_path", "image_count", "matches", "suggested_confidence", "status", "notes"]
    with args.group_review_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        for row in group_records:
            group_id = str(row.pop("group_id"))
            old = old_groups.get(group_id, {})
            row["candidate_id"] = group_id
            if old.get("status") in {"approved", "rejected"}:
                row["status"], row["notes"] = old["status"], old["notes"]
            writer.writerow(row)
    group_display = [{
        "image_path": row["representative_path"],
        "display_label": f"{row['candidate_id']} n={row['image_count']} {row['status']}",
        "display_detail": str(row["matches"]),
    } for row in group_records]
    write_contact_sheet(group_display, args.output_dir / "cannon_evolution_review_groups.jpg",
                        f"Condensed Cannon Evolution review - {len(group_records)} groups", len(group_records))

    existing = load_existing_reviews(args.review_file)
    review_fields = ["candidate_id", "image_path", "match", "slot", "timestamp_ms", "status", "notes"]
    args.review_file.parent.mkdir(parents=True, exist_ok=True)
    with args.review_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in records:
            old = existing.get(str(row["candidate_id"]), {})
            writer.writerow({
                "candidate_id": row["candidate_id"], "image_path": row["image_path"],
                "match": row["match"], "slot": row["slot"],
                "timestamp_ms": row["timestamp_ms"],
                "status": old.get("status", "uncertain"), "notes": old.get("notes", ""),
            })

    summary = {
        "matches": args.matches,
        "searched_image_count": len(feature_by_path),
        "candidate_count": len(records),
        "normal_cannon_reference_count": len(cannon_bank),
        "thresholds": {
            "min_cannon_similarity": args.min_cannon_similarity,
            "min_confidence": args.min_confidence,
        },
        "outputs": {
            "candidates_csv": str(candidates_path),
            "candidate_contact_sheet": str(args.output_dir / "cannon_evolution_candidates.jpg"),
            "manual_review": str(args.review_file),
            "condensed_manual_review": str(args.group_review_file),
            "group_members": str(membership_path),
        },
        "automatic_labeling": False,
    }
    (args.output_dir / "evolution_discovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Images searched: {len(feature_by_path)}")
    print(f"Candidates:      {len(records)}")
    print(f"Candidate CSV:   {candidates_path}")
    print(f"Manual review:   {args.review_file}")
    print(f"Review groups:   {len(group_records)} in {args.group_review_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
