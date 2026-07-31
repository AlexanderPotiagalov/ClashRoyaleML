from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np


TIMESTAMP_RE = re.compile(r"_t(\d+)ms(?:\.[^.]+)?$")
NORMAL_CLASSES = (
    "electro_spirit", "fireball", "barbarian_barrel", "knight",
    "hero_musketeer", "hog_rider", "cannon", "skeletons",
)
VISUAL_CLASSES = (*NORMAL_CLASSES, "cannon_evolution", "empty", "unknown")


def timestamp_ms(path_or_name: str | Path) -> int:
    name = Path(path_or_name).name
    match = TIMESTAMP_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse timestamp from: {name}")
    return int(match.group(1))


def discover_slot_images(match_dir: Path) -> list[Path]:
    if not match_dir.is_dir():
        raise FileNotFoundError(f"Missing card-slot directory: {match_dir}")
    paths: list[Path] = []
    for slot_number in range(1, 5):
        slot = match_dir / f"slot_{slot_number}"
        if not slot.is_dir():
            raise FileNotFoundError(f"Missing slot directory: {slot}")
        paths.extend(sorted(slot.glob("*.jpg"), key=timestamp_ms))
    if not paths:
        raise ValueError(f"No JPG card slots found under: {match_dir}")
    return paths


def unit(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def visual_embedding(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    artwork = resized[18:101, 12:84]
    small = cv2.resize(artwork, (18, 22), cv2.INTER_AREA)
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    gray = gray - gray.mean()
    gray /= max(float(gray.std()), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = (lab[:, :, 1:] - 128.0) / 64.0
    return unit(np.concatenate([2.5 * gray.ravel(), gx.ravel(), gy.ravel(), 0.8 * chroma.ravel()]))


def visual_stats(image: np.ndarray) -> dict[str, float]:
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    banner = hsv[:37, 18:78]
    border_mask = np.zeros(hsv.shape[:2], bool)
    border_mask[:, :8] = True
    border_mask[:, -8:] = True
    border_mask[-20:] = True
    purple = (
        (hsv[:, :, 0] >= 125) & (hsv[:, :, 0] <= 175)
        & (hsv[:, :, 1] > 80) & border_mask
    )
    orange_banner = (
        (banner[:, :, 0] >= 5) & (banner[:, :, 0] <= 35)
        & (banner[:, :, 1] > 100) & (banner[:, :, 2] > 120)
    )
    return {
        "brightness": float(hsv[:, :, 2].mean()),
        "saturation": float(hsv[:, :, 1].mean()),
        "purple_border_fraction": float(purple.mean()),
        "orange_banner_fraction": float(orange_banner.mean()),
    }


def is_empty(image: np.ndarray) -> bool:
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    center = hsv[30:105, 18:78]
    blue = (
        (center[:, :, 0] >= 90) & (center[:, :, 0] <= 125)
        & (center[:, :, 1] >= 90)
    )
    texture = cv2.Laplacian(center[:, :, 2], cv2.CV_32F).var()
    return bool(blue.mean() > 0.72 and texture < 220)


def load_normal_reference_paths(labels_csv: Path) -> dict[str, list[Path]]:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing match_001 labels: {labels_csv}")
    grouped = {label: [] for label in NORMAL_CLASSES}
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label", "recovered"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid labels CSV: {labels_csv}")
        for row in reader:
            if row["label"] in grouped and row["recovered"].lower() != "true":
                path = Path(row["image_path"])
                if path.is_file():
                    grouped[row["label"]].append(path)
    missing = [label for label, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Normal reference classes have no images: {missing}")
    return grouped


def build_reference_banks(
    grouped: dict[str, list[Path]], limit: int = 160
) -> tuple[dict[str, np.ndarray], dict[str, list[Path]]]:
    banks: dict[str, np.ndarray] = {}
    selected_paths: dict[str, list[Path]] = {}
    for label, paths in grouped.items():
        ordered = sorted(paths, key=lambda path: str(path))
        if len(ordered) > limit:
            positions = np.linspace(0, len(ordered) - 1, limit).astype(int)
            ordered = [ordered[position] for position in positions]
        features: list[np.ndarray] = []
        valid: list[Path] = []
        for path in ordered:
            image = cv2.imread(str(path))
            if image is not None:
                features.append(visual_embedding(image))
                valid.append(path)
        if not features:
            raise ValueError(f"Could not read reference images for {label}")
        banks[label] = np.vstack(features)
        selected_paths[label] = valid
    return banks, selected_paths


def bank_similarity(feature: np.ndarray, bank: np.ndarray, neighbours: int = 3) -> float:
    similarities = np.sort(bank @ feature)[::-1]
    return float(similarities[: min(neighbours, len(similarities))].mean())


def score_classes(feature: np.ndarray, banks: dict[str, np.ndarray]) -> list[tuple[float, str]]:
    return sorted(
        ((bank_similarity(feature, bank), label) for label, bank in banks.items()),
        reverse=True,
    )


def write_contact_sheet(
    rows: list[dict[str, object]], output: Path, title: str, limit: int = 48
) -> None:
    selected = rows[:limit]
    columns, tile_width, tile_height, header = 6, 164, 220, 48
    row_count = max(1, int(np.ceil(len(selected) / columns)))
    sheet = np.full((header + row_count * tile_height, columns * tile_width, 3), 28, np.uint8)
    cv2.putText(sheet, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    for position, row in enumerate(selected):
        path = Path(str(row["image_path"]))
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(142 / w, 160 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), cv2.INTER_AREA)
        grid_row, column = divmod(position, columns)
        x = column * tile_width + (tile_width - image.shape[1]) // 2
        y = header + grid_row * tile_height + 3
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        primary = str(row.get("display_label", row.get("visual_label", "candidate")))
        secondary_value = row.get("display_detail")
        if secondary_value is None:
            try:
                secondary_value = f"{path.parent.name} {timestamp_ms(path)}ms"
            except ValueError:
                secondary_value = path.name
        secondary = str(secondary_value)
        cv2.putText(sheet, primary[:24], (column * tile_width + 4, y + 179),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, secondary[:27], (column * tile_width + 4, y + 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output}")
