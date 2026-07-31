from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

try:
    from card_reference_utils import timestamp_ms, write_contact_sheet
except ModuleNotFoundError:  # Supports `python -m scripts...` and test imports.
    from scripts.card_reference_utils import timestamp_ms, write_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Cannon Evolution reference set from manually approved candidates."
    )
    parser.add_argument(
        "--review-file", type=Path, default=Path("config/cannon_evolution_group_review.csv")
    )
    parser.add_argument(
        "--group-members", type=Path,
        default=Path("data/card_clusters/evolution_discovery/evolution_candidate_group_members.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/card_dataset/reference/cannon_evolution"),
    )
    parser.add_argument(
        "--seed-image", type=Path,
        default=Path("data/card_dataset/reference/cannon_evolution/evo_cannon_reference.png"),
        help="Optional user-confirmed canonical Evo Cannon image",
    )
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--contact-examples", type=int, default=72)
    return parser.parse_args()


def materialize(source: Path, destination: Path, copy_only: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing reference differs: {destination}")
        try:
            return "existing_hardlink" if os.path.samefile(source, destination) else "existing_file"
        except OSError:
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
    if args.contact_examples < 1:
        raise ValueError("--contact-examples must be positive")
    if not args.review_file.is_file():
        raise FileNotFoundError(f"Missing review file: {args.review_file}")
    with args.review_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "status"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid review file: {args.review_file}")
        rows = [dict(row) for row in reader]
    invalid = [row for row in rows if row["status"] not in {"approved", "rejected", "uncertain"}]
    if invalid:
        raise ValueError(f"Invalid review status: {invalid[0]['status']!r}")
    approved = [row for row in rows if row["status"] == "approved"]
    if approved and "image_path" not in rows[0]:
        if not args.group_members.is_file():
            raise FileNotFoundError(f"Missing candidate group membership: {args.group_members}")
        with args.group_members.open(newline="", encoding="utf-8") as handle:
            members = list(csv.DictReader(handle))
        approved_ids = {row["candidate_id"] for row in approved}
        approved = [
            {**member, "status": "approved", "notes": "approved via visual group"}
            for member in members if member["group_id"] in approved_ids
        ]
    if args.seed_image.is_file():
        approved.append({
            "candidate_id": "confirmed_seed",
            "image_path": str(args.seed_image.resolve()),
            "match": "reference",
            "slot": "reference",
            "timestamp_ms": "0",
            "status": "approved",
            "notes": "user-confirmed canonical Cannon Evolution reference",
        })
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for row in approved:
        source = Path(row["image_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Approved image is missing: {source}")
        destination = args.output / f"{row['match']}__{row['slot']}__{source.name}"
        method = materialize(source.resolve(), destination, args.copy_only)
        manifest_rows.append({
            **row, "reference_path": str(destination.resolve()), "link_method": method,
        })
    fields = [
        "candidate_id", "image_path", "reference_path", "match", "slot",
        "timestamp_ms", "status", "notes", "link_method",
    ]
    manifest = args.output / "approved_examples.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)
    display = [
        {
            "image_path": row["reference_path"],
            "display_label": "cannon_evolution approved",
            "display_detail": f"{row['match']} {row['slot']} {row['timestamp_ms']}ms",
        }
        for row in manifest_rows
    ]
    sheet = args.output / "approved_cannon_evolution.jpg"
    write_contact_sheet(display, sheet, f"Approved Cannon Evolution - {len(display)}", args.contact_examples)
    summary = {
        "review_file": str(args.review_file),
        "reviewed_candidate_count": len(rows),
        "approved_count": len(approved),
        "rejected_count": sum(row["status"] == "rejected" for row in rows),
        "uncertain_count": sum(row["status"] == "uncertain" for row in rows),
        "manifest": str(manifest),
        "contact_sheet": str(sheet),
    }
    (args.output / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Approved references: {len(approved)}")
    print(f"Reference directory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
