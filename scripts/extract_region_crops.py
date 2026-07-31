from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export crops from replay frames using saved ROI coordinates."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/frames/match_001"),
        help="Directory containing extracted JPG replay frames",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("config/regions_720x1612.json"),
        help="ROI configuration JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crops/match_001"),
        help="Output directory for cropped datasets",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of region names to export",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite crops that already exist",
    )
    return parser.parse_args()


def load_regions(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Region configuration not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    regions = config.get("regions")

    if not isinstance(regions, dict) or not regions:
        raise ValueError(f"No regions found in {path}")

    return regions


def normalized_to_pixels(
    normalized: dict[str, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round(float(normalized["x1"]) * image_width))
    y1 = int(round(float(normalized["y1"]) * image_height))
    x2 = int(round(float(normalized["x2"]) * image_width))
    y2 = int(round(float(normalized["y2"]) * image_height))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


def main() -> int:
    args = parse_args()

    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    all_regions = load_regions(args.regions)

    if args.only:
        unknown = sorted(set(args.only) - set(all_regions))
        if unknown:
            raise ValueError(
                "Unknown region name(s): "
                + ", ".join(unknown)
                + "\nAvailable: "
                + ", ".join(sorted(all_regions))
            )
        selected_regions = {
            name: all_regions[name]
            for name in args.only
        }
    else:
        selected_regions = all_regions

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No JPG frames found in: {args.frames_dir}")

    args.output.mkdir(parents=True, exist_ok=True)

    for region_name in selected_regions:
        (args.output / region_name).mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / "manifest.jsonl"
    exported_count = 0
    skipped_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for frame_path in tqdm(frame_paths, desc="Exporting region crops", unit="frame"):
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"WARNING: could not read {frame_path}; skipping")
                continue

            image_height, image_width = image.shape[:2]

            manifest_row = {
                "source_frame": str(frame_path.resolve()),
                "frame_name": frame_path.name,
                "frame_width": image_width,
                "frame_height": image_height,
                "crops": {},
            }

            for region_name, region_data in selected_regions.items():
                normalized = region_data.get("normalized")
                if not isinstance(normalized, dict):
                    raise ValueError(
                        f"Region '{region_name}' has no normalized coordinates"
                    )

                x1, y1, x2, y2 = normalized_to_pixels(
                    normalized,
                    image_width,
                    image_height,
                )

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(
                        f"Empty crop for region '{region_name}' in {frame_path}"
                    )

                output_path = args.output / region_name / frame_path.name

                if output_path.exists() and not args.overwrite:
                    skipped_count += 1
                else:
                    written = cv2.imwrite(
                        str(output_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                    )
                    if not written:
                        raise RuntimeError(f"Failed to save {output_path}")
                    exported_count += 1

                manifest_row["crops"][region_name] = {
                    "path": str(output_path.resolve()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }

            manifest.write(json.dumps(manifest_row) + "\n")

    print()
    print(f"Source frames:       {len(frame_paths)}")
    print(f"Selected regions:    {len(selected_regions)}")
    print(f"New crops exported:  {exported_count}")
    print(f"Existing crops kept: {skipped_count}")
    print(f"Output directory:    {args.output}")
    print(f"Manifest:            {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())