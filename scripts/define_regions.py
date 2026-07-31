from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


REGIONS = [
    ("arena", "Select the full battlefield / arena"),
    ("hand", "Select the four cards currently in hand"),
    ("next_card", "Select the next-card preview"),
    ("elixir", "Select the elixir bar / number"),
    ("timer", "Select the match timer"),
    ("friendly_left_tower", "Select the friendly left Princess Tower"),
    ("friendly_king_tower", "Select the friendly King Tower"),
    ("friendly_right_tower", "Select the friendly right Princess Tower"),
    ("enemy_left_tower", "Select the enemy left Princess Tower"),
    ("enemy_king_tower", "Select the enemy King Tower"),
    ("enemy_right_tower", "Select the enemy right Princess Tower"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively define Clash Royale regions of interest."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/frames/inspection/first_frame.jpg"),
        help="Input frame used to define regions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/regions_720x1612.json"),
        help="Output JSON configuration",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("data/frames/inspection/regions_preview.jpg"),
        help="Output image showing all selected regions",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=650,
        help="Maximum popup width in pixels",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=850,
        help="Maximum popup height in pixels",
    )
    return parser.parse_args()


def normalize_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    return {
        "x1": x / image_width,
        "y1": y / image_height,
        "x2": (x + width) / image_width,
        "y2": (y + height) / image_height,
    }


def fit_image(image, max_width: int, max_height: int):
    image_height, image_width = image.shape[:2]
    scale = min(
        max_width / image_width,
        max_height / image_height,
        1.0,
    )

    if scale >= 1.0:
        return image.copy(), 1.0

    display_width = max(1, int(round(image_width * scale)))
    display_height = max(1, int(round(image_height * scale)))

    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"OpenCV could not read: {args.image}")

    image_height, image_width = image.shape[:2]

    display_image, display_scale = fit_image(
        image,
        max_width=args.max_width,
        max_height=args.max_height,
    )

    display_height, display_width = display_image.shape[:2]

    print()
    print("ROI CONTROLS")
    print("- Drag a rectangle with the mouse.")
    print("- Press ENTER or SPACE to accept.")
    print("- Press C to skip the current region.")
    print("- Press ESC or close the window to stop.")
    print()
    print(
        f"Original frame: {image_width}x{image_height} | "
        f"Popup: {display_width}x{display_height} | "
        f"Scale: {display_scale:.3f}"
    )
    print()
    print("Tower orientation is from your point of view.")
    print()

    regions: dict[str, dict[str, object]] = {}

    for region_name, instruction in REGIONS:
        window_title = f"{region_name}: {instruction}"
        print(f"Selecting '{region_name}': {instruction}")

        display_x, display_y, display_w, display_h = cv2.selectROI(
            window_title,
            display_image,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow(window_title)

        display_x = int(display_x)
        display_y = int(display_y)
        display_w = int(display_w)
        display_h = int(display_h)

        if display_w <= 0 or display_h <= 0:
            print(f"Skipped '{region_name}'.")
            continue

        x = int(round(display_x / display_scale))
        y = int(round(display_y / display_scale))
        width = int(round(display_w / display_scale))
        height = int(round(display_h / display_scale))

        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))

        regions[region_name] = {
            "pixels": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "normalized": normalize_box(
                x,
                y,
                width,
                height,
                image_width,
                image_height,
            ),
        }

    cv2.destroyAllWindows()

    if not regions:
        raise RuntimeError("No regions were selected.")

    config = {
        "source_image": str(args.image),
        "frame_width": image_width,
        "frame_height": image_height,
        "display_scale": display_scale,
        "regions": regions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    preview = image.copy()
    palette = [
        (255, 255, 255),
        (255, 180, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 100, 100),
        (100, 100, 255),
        (120, 220, 255),
        (255, 120, 220),
        (80, 200, 255),
        (220, 120, 255),
    ]

    for index, (region_name, region_data) in enumerate(regions.items()):
        pixels = region_data["pixels"]

        x = int(pixels["x"])
        y = int(pixels["y"])
        width = int(pixels["width"])
        height = int(pixels["height"])
        colour = palette[index % len(palette)]

        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            colour,
            3,
        )

        label_y = max(25, y - 8)
        cv2.putText(
            preview,
            region_name,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

    args.preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.preview), preview):
        raise RuntimeError(
            f"Could not save preview to: {args.preview}"
        )

    print()
    print(f"Saved configuration: {args.output}")
    print(f"Saved preview:       {args.preview}")
    print(f"Selected regions:    {len(regions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())