from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
from tqdm import tqdm


TIMESTAMP_PATTERN = re.compile(r"_t(\d+)ms", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split each Clash Royale hand crop into four card-slot images."
    )
    parser.add_argument(
        "--hand-dir",
        type=Path,
        default=Path("data/crops/match_001/hand"),
        help="Directory containing full four-card hand crops",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/card_slots/match_001"),
        help="Output directory for slot_1 through slot_4",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
    )
    parser.add_argument(
        "--inner-trim",
        type=int,
        default=1,
        help="Pixels removed from each slot's left and right edges",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing slot images",
    )
    return parser.parse_args()


def extract_timestamp_ms(filename: str) -> int | None:
    match = TIMESTAMP_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def calculate_slot_bounds(
    image_width: int,
    slot_index: int,
    inner_trim: int,
) -> tuple[int, int]:
    if slot_index not in {0, 1, 2, 3}:
        raise ValueError("slot_index must be from 0 to 3")

    left = round(image_width * slot_index / 4)
    right = round(image_width * (slot_index + 1) / 4)

    left += inner_trim
    right -= inner_trim

    if right <= left:
        raise ValueError(
            "The slot became empty. Reduce --inner-trim."
        )

    return left, right


def main() -> int:
    args = parse_args()

    if not args.hand_dir.exists():
        raise FileNotFoundError(
            f"Hand-crop directory not found: {args.hand_dir}"
        )

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    if args.inner_trim < 0:
        raise ValueError("--inner-trim cannot be negative")

    hand_paths = sorted(args.hand_dir.glob("*.jpg"))
    if not hand_paths:
        raise RuntimeError(
            f"No JPG hand crops found in: {args.hand_dir}"
        )

    slot_dirs = []
    for slot_number in range(1, 5):
        slot_dir = args.output / f"slot_{slot_number}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_dirs.append(slot_dir)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    preview_path = args.output / "slot_preview.jpg"

    exported = 0
    skipped = 0
    preview_written = False

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for hand_path in tqdm(
            hand_paths,
            desc="Splitting card slots",
            unit="hand",
        ):
            image = cv2.imread(str(hand_path))
            if image is None:
                print(f"WARNING: could not read {hand_path}; skipping")
                continue

            height, width = image.shape[:2]
            row = {
                "source_hand_crop": str(hand_path.resolve()),
                "frame_name": hand_path.name,
                "timestamp_ms": extract_timestamp_ms(hand_path.name),
                "hand_width": width,
                "hand_height": height,
                "slots": {},
            }

            preview = image.copy()

            for slot_index in range(4):
                slot_number = slot_index + 1
                left, right = calculate_slot_bounds(
                    width,
                    slot_index,
                    args.inner_trim,
                )

                slot_image = image[:, left:right]
                if slot_image.size == 0:
                    raise RuntimeError(
                        f"Empty slot {slot_number} in {hand_path}"
                    )

                output_path = (
                    slot_dirs[slot_index] / hand_path.name
                )

                if output_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    written = cv2.imwrite(
                        str(output_path),
                        slot_image,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                    )
                    if not written:
                        raise RuntimeError(
                            f"Could not save: {output_path}"
                        )
                    exported += 1

                row["slots"][f"slot_{slot_number}"] = {
                    "path": str(output_path.resolve()),
                    "x1": left,
                    "y1": 0,
                    "x2": right,
                    "y2": height,
                    "width": right - left,
                    "height": height,
                }

                cv2.rectangle(
                    preview,
                    (left, 0),
                    (right - 1, height - 1),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    str(slot_number),
                    (left + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if not preview_written:
                if not cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError(
                        f"Could not save preview: {preview_path}"
                    )
                preview_written = True

            manifest.write(json.dumps(row) + "\n")

    print()
    print(f"Hand crops processed: {len(hand_paths)}")
    print(f"New slot images:      {exported}")
    print(f"Existing images kept: {skipped}")
    print(f"Output directory:     {args.output}")
    print(f"Manifest:             {manifest_path}")
    print(f"Preview:              {preview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
