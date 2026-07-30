from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
from tqdm import tqdm


def fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Crop values must be between 0 and 1.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export timestamped frames from a replay video."
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 replay")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for JPG frames and manifest.jsonl",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Number of frames to export per second (default: 2)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality from 1 to 100 (default: 92)",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=fraction,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Normalized crop fractions; default exports the full frame",
    )
    return parser.parse_args()


def validate_crop(crop: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, crop)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop must satisfy X2 > X1 and Y2 > Y1.")
    return x1, y1, x2, y2


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be from 1 to 100.")

    crop = validate_crop(args.crop)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        capture.release()
        raise RuntimeError("The video reported an invalid FPS value.")

    sample_interval = 1.0 / args.sample_fps
    next_sample_time = 0.0
    exported = 0

    x1 = int(round(crop[0] * source_width))
    y1 = int(round(crop[1] * source_height))
    x2 = int(round(crop[2] * source_width))
    y2 = int(round(crop[3] * source_height))

    with manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(total=total_frames, unit="frame", desc="Reading video")
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_seconds = frame_index / source_fps

            if timestamp_seconds + 1e-9 >= next_sample_time:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    capture.release()
                    progress.close()
                    raise RuntimeError("Crop produced an empty image.")

                timestamp_ms = int(round(timestamp_seconds * 1000))
                filename = f"frame_{exported:06d}_t{timestamp_ms:09d}ms.jpg"
                frame_path = args.output / filename

                written = cv2.imwrite(
                    str(frame_path),
                    cropped,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not written:
                    capture.release()
                    progress.close()
                    raise RuntimeError(f"Failed to write {frame_path}")

                row = {
                    "frame_id": exported,
                    "source_video": str(args.video.resolve()),
                    "source_frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "image_path": str(frame_path.resolve()),
                    "source_width": source_width,
                    "source_height": source_height,
                    "output_width": int(cropped.shape[1]),
                    "output_height": int(cropped.shape[0]),
                    "crop": {
                        "x1": crop[0],
                        "y1": crop[1],
                        "x2": crop[2],
                        "y2": crop[3],
                    },
                }
                manifest.write(json.dumps(row) + "\n")
                exported += 1
                next_sample_time += sample_interval

            frame_index += 1
            progress.update(1)

        progress.close()
        capture.release()

    if exported == 0:
        raise RuntimeError("No frames were exported.")

    print(f"Exported {exported} frames to: {args.output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
