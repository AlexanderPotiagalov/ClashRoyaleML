from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a replay video.")
    parser.add_argument("video", type=Path, help="Path to an MP4 replay")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/frames/inspection/first_frame.jpg"),
        help="Where to save the first frame",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {args.video}. "
            "Confirm it is a normal MP4 and is not corrupted."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0.0

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("The video opened, but the first frame could not be read.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"Could not save first frame to {args.output}")

    print(f"Video: {args.video}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"First frame saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
