from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CANONICAL_LABELS = {
    0: "barbarian_barrel",
    1: "knight",
    2: "fireball",
    3: "empty",
    4: "hog_rider",
    5: "cannon",
    6: "electro_spirit",
    7: "skeletons",
    8: "hero_musketeer",
}

CARD_COSTS = {0: 2, 1: 3, 2: 4, 4: 4, 5: 3, 6: 1, 7: 1, 8: 4}
REFERENCE_IMAGES = {
    0: ("slot_2", "frame_000421_t000210507ms.jpg"),
    1: ("slot_3", "frame_000583_t000291512ms.jpg"),
    2: ("slot_2", "frame_000062_t000031013ms.jpg"),
    4: ("slot_3", "frame_000058_t000029016ms.jpg"),
    5: ("slot_1", "frame_000381_t000190504ms.jpg"),
    6: ("slot_3", "frame_000295_t000147510ms.jpg"),
    7: ("slot_2", "frame_000284_t000142005ms.jpg"),
    8: ("slot_1", "frame_000540_t000270015ms.jpg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Clash Royale card-slot crops into visual groups."
    )
    parser.add_argument(
        "--slots-dir",
        type=Path,
        default=Path("data/card_slots/match_001"),
        help="Directory containing slot_1 through slot_4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/card_clusters/match_001"),
        help="Output directory for cluster assignments and contact sheets",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=9,
        help="Number of final card classes (9 enables stable deck labels)",
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=24,
        help="Number of representative images shown in each contact sheet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for OpenCV k-means",
    )
    return parser.parse_args()


def discover_images(slots_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for slot_number in range(1, 5):
        slot_dir = slots_dir / f"slot_{slot_number}"
        if not slot_dir.exists():
            raise FileNotFoundError(f"Missing slot directory: {slot_dir}")
        paths.extend(sorted(slot_dir.glob("*.jpg")))

    if not paths:
        raise RuntimeError(f"No slot JPGs found under: {slots_dir}")
    return paths


def _normalise_block(block: np.ndarray, weight: float) -> np.ndarray:
    block = block.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(block))
    if norm > 1e-8:
        block /= norm
    return block * weight


def card_feature(image: np.ndarray) -> np.ndarray:
    """A spatial colour/shape descriptor including evolution UI and borders."""
    height, width = image.shape[:2]
    image = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)

    # Artwork is intentionally retained at a higher resolution than before.
    artwork = image[8:108, 7:89]
    lab = cv2.cvtColor(
        cv2.resize(artwork, (24, 30), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    lab = (lab - np.array([128.0, 128.0, 128.0], np.float32)) / 128.0

    gray = cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 60), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_parts: list[np.ndarray] = []
    for row in range(0, 60, 10):
        for column in range(0, 48, 8):
            hist, _ = np.histogram(
                angle[row:row + 10, column:column + 8],
                bins=8,
                range=(0.0, 360.0),
                weights=magnitude[row:row + 10, column:column + 8],
            )
            hog_parts.append(hist.astype(np.float32))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    spatial_histograms: list[np.ndarray] = []
    for y1, y2 in ((4, 43), (43, 84), (84, 123)):
        for x1, x2 in ((5, 48), (48, 91)):
            histogram = cv2.calcHist(
                [hsv[y1:y2, x1:x2]], [0, 1], None, [12, 4], [0, 180, 0, 256]
            )
            spatial_histograms.append(histogram.flatten())

    # Evolution diamonds and the animated card border occupy few pixels. Giving
    # them their own spatial blocks stops common artwork from drowning them out.
    evolution_banner = cv2.resize(
        image[:35, 20:76], (28, 18), interpolation=cv2.INTER_AREA
    )
    evolution_banner = cv2.cvtColor(evolution_banner, cv2.COLOR_BGR2LAB)
    border = np.concatenate(
        [
            cv2.resize(image[:, :8], (8, 24), interpolation=cv2.INTER_AREA),
            cv2.resize(image[:, -8:], (8, 24), interpolation=cv2.INTER_AREA),
            cv2.resize(image[-20:, 8:-8], (16, 12), interpolation=cv2.INTER_AREA),
        ],
        axis=None,
    )

    return np.concatenate(
        [
            _normalise_block(lab, 0.9),
            _normalise_block(np.concatenate(hog_parts), 3.2),
            _normalise_block(np.concatenate(spatial_histograms), 0.7),
            _normalise_block(evolution_banner, 4.2),
            _normalise_block(border, 2.2),
        ]
    ).astype(np.float32)


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly scale dimensions without letting transition frames set the scale."""
    center = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    scale = q75 - q25
    fallback = features.std(axis=0)
    scale = np.where(scale < 1e-5, fallback, scale)
    scale = np.where(scale < 1e-5, 1.0, scale)
    standardized = np.clip((features - center) / scale, -8.0, 8.0)
    return standardized.astype(np.float32), center, scale


def reduce_features(features: np.ndarray, dimensions: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Use a reproducible random projection to retain distances inexpensively."""
    dimensions = min(dimensions, features.shape[1])
    generator = np.random.default_rng(1729)
    projection = generator.choice(
        np.array([-1.0, 0.0, 1.0], np.float32),
        size=(features.shape[1], dimensions),
        p=(1 / 6, 2 / 3, 1 / 6),
    )
    projection *= np.sqrt(3.0 / dimensions)
    return (features @ projection).astype(np.float32), projection


def empty_slot_mask(images: list[np.ndarray]) -> np.ndarray:
    """Detect the stable blue crown placeholder; it is unknown, not a card."""
    result = np.zeros(len(images), dtype=bool)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        center = hsv[30:105, 18:78]
        blue = (
            (center[:, :, 0] >= 90)
            & (center[:, :, 0] <= 125)
            & (center[:, :, 1] >= 90)
        )
        texture = cv2.Laplacian(center[:, :, 2], cv2.CV_32F).var()
        result[index] = blue.mean() > 0.72 and texture < 180.0
    return result


def transition_frame_mask(images: list[np.ndarray]) -> np.ndarray:
    """Catch coherent end-screen/chat overlays that are not isolated in feature space."""
    result = np.zeros(len(images), dtype=bool)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        side_edges = np.concatenate(
            [edges[20:120, 2:15].ravel(), edges[20:120, 81:94].ravel()]
        ).mean() / 255.0
        white = np.all(resized > 220, axis=2).mean()
        result[index] = white > 0.25 and side_edges < 0.15
    return result


def coherent_outlier_mask(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flag isolated transition frames while preserving small, coherent variants."""
    # Fifth-neighbour distance is small for a repeated rare state, but large for
    # one-off fades/overlays. Work in PCA space to make the O(n^2) pass cheap.
    sample = features[:, : min(48, features.shape[1])]
    squared = np.sum(sample * sample, axis=1)
    distances = squared[:, None] + squared[None, :] - 2.0 * sample @ sample.T
    np.maximum(distances, 0.0, out=distances)
    neighbour_distance = np.sqrt(np.partition(distances, 5, axis=1)[:, 5])
    median = float(np.median(neighbour_distance))
    mad = float(np.median(np.abs(neighbour_distance - median)))
    threshold = median + 7.0 * max(mad, 1e-6)
    return neighbour_distance > threshold, neighbour_distance


def smooth_single_frame_flickers(labels: np.ndarray, paths: list[Path]) -> np.ndarray:
    """Remove impossible one-frame class changes within an individual hand slot."""
    smoothed = labels.copy()
    for slot_name in ("slot_1", "slot_2", "slot_3", "slot_4"):
        indices = [i for i, path in enumerate(paths) if path.parent.name == slot_name]
        for position, index in enumerate(indices[1:-1], start=1):
            previous = indices[position - 1]
            following = indices[position + 1]
            if (
                smoothed[index] >= 0
                and smoothed[previous] == smoothed[following]
                and smoothed[previous] >= 0
                and smoothed[index] != smoothed[previous]
            ):
                # A one-frame disagreement is normally a hand-animation blend.
                # Reject it rather than copying a neighbouring label onto it.
                smoothed[index] = -1
    return smoothed


def identity_feature(image: np.ndarray) -> np.ndarray:
    """Describe card artwork while suppressing colour and availability overlays."""
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    artwork = resized[18:101, 12:84]
    gray = cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    pixels = cv2.resize(gray, (18, 22), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=3)
    vector = np.concatenate([pixels.ravel(), 0.6 * gx.ravel(), 0.6 * gy.ravel()])
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def badge_feature(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    # Tight crop around the white cost digit; excluding the coloured badge
    # background makes 1/2/3/4 template matching much more reliable.
    badge = resized[102:124, 39:59]
    badge = cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY)
    badge = cv2.equalizeHist(badge)
    badge = cv2.resize(badge, (14, 14), interpolation=cv2.INTER_AREA).astype(np.float32)
    badge -= badge.mean()
    return badge.ravel() / max(float(np.linalg.norm(badge)), 1e-8)


def prototype_labels(
    images: list[np.ndarray],
    paths: list[Path],
    card_indices: np.ndarray,
    reject_margin: float = 0.035,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify against verified card anchors; reject ambiguous edge frames."""
    identities = np.vstack([identity_feature(image) for image in images])
    badges = np.vstack([badge_feature(image) for image in images])
    prototype_sets: dict[int, list[int]] = {label: [] for label in CANONICAL_LABELS}
    anchor_indices: dict[int, int] = {}
    for semantic_id, (slot_name, filename) in REFERENCE_IMAGES.items():
        anchor = next(
            index for index, path in enumerate(paths)
            if path.parent.name == slot_name and path.name == filename
        )
        anchor_indices[semantic_id] = anchor
        prototype_sets[semantic_id] = [anchor]

    labels = np.full(len(images), -1, dtype=np.int32)
    confidence = np.zeros(len(images), dtype=np.float32)
    semantic_ids = sorted(
        semantic_id for semantic_id, indices in prototype_sets.items() if indices
    )
    for index in card_indices:
        cost_scores: dict[int, float] = {}
        for cost in sorted(set(CARD_COSTS.values())):
            cost_prototypes = [
                anchor_indices[semantic_id]
                for semantic_id in anchor_indices
                if CARD_COSTS.get(semantic_id) == cost
            ]
            badge_distance = 1.0 - badges[cost_prototypes] @ badges[index]
            cost_scores[cost] = float(np.sort(badge_distance)[:3].mean())
        predicted_cost = min(cost_scores, key=cost_scores.get)
        if cost_scores[predicted_cost] > 0.48:
            continue
        candidates = [
            semantic_id
            for semantic_id in semantic_ids
            if CARD_COSTS[semantic_id] == predicted_cost
        ]
        scores: list[float] = []
        for semantic_id in candidates:
            prototype_indices = prototype_sets[semantic_id]
            similarity = identities[prototype_indices] @ identities[index]
            nearest = np.sort(1.0 - similarity)[: min(3, len(similarity))]
            scores.append(float(nearest.mean()))
        order = np.argsort(scores)
        best = scores[order[0]]
        second = scores[order[1]] if len(order) > 1 else best + 1.0
        margin = (second - best) / max(second, 1e-6)
        confidence[index] = margin
        if margin >= reject_margin and best <= 0.48:
            labels[index] = candidates[order[0]]
    return labels, confidence


def contact_sheet(
    examples: list[tuple[Path, str]],
    title: str,
    output_path: Path,
    tile_width: int = 160,
    tile_height: int = 206,
    columns: int = 6,
) -> None:
    if not examples:
        return
    rows = int(np.ceil(len(examples) / columns))
    header_height = 48
    sheet = np.full(
        (header_height + rows * tile_height, columns * tile_width, 3),
        30,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet, title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    for index, (image_path, label) in enumerate(examples):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        max_w, max_h = tile_width - 14, tile_height - 44
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        display_w, display_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * tile_width + (tile_width - display_w) // 2
        y = header_height + row * tile_height + 3
        sheet[y:y + display_h, x:x + display_w] = resized
        cv2.putText(
            sheet, image_path.parent.name,
            (column * tile_width + 5, header_height + (row + 1) * tile_height - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, label[:24],
            (column * tile_width + 5, header_height + (row + 1) * tile_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 220, 255), 1, cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not save contact sheet: {output_path}")


def main() -> int:
    args = parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")
    if args.examples_per_cluster < 1:
        raise ValueError("--examples-per-cluster must be positive")

    image_paths = discover_images(args.slots_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    # Avoid leaving convincing but stale sheets behind after --clusters changes.
    for stale_sheet in args.output.glob("cluster_[0-9][0-9].jpg"):
        stale_sheet.unlink()
    print(f"Found {len(image_paths)} card-slot images.")

    features: list[np.ndarray] = []
    valid_paths: list[Path] = []
    images: list[np.ndarray] = []
    for path in tqdm(image_paths, desc="Extracting card features", unit="image"):
        image = cv2.imread(str(path))
        if image is None:
            print(f"WARNING: could not read {path}; skipping")
            continue
        features.append(card_feature(image))
        valid_paths.append(path)
        images.append(image)

    canonical_output = args.clusters in (9, 10)
    raw_cluster_count = 10 if canonical_output else args.clusters
    if len(features) < raw_cluster_count:
        raise RuntimeError(
            f"Only {len(features)} valid images for {raw_cluster_count} raw clusters."
        )

    feature_matrix = np.vstack(features)
    # Each descriptor block is already L2-normalised and deliberately weighted.
    # Scaling individual pixels here would erase those semantic weights.
    feature_center = np.zeros(feature_matrix.shape[1], dtype=np.float32)
    feature_scale = np.ones(feature_matrix.shape[1], dtype=np.float32)
    reduced, projection = reduce_features(feature_matrix)
    reduced_center = np.zeros(reduced.shape[1], dtype=np.float32)
    reduced_scale = np.ones(reduced.shape[1], dtype=np.float32)

    empty = empty_slot_mask(images)
    transition = transition_frame_mask(images)
    isolated, outlier_score = coherent_outlier_mask(reduced)
    unknown = empty | transition | isolated
    card_indices = np.flatnonzero(~unknown)
    if len(card_indices) < raw_cluster_count:
        raise RuntimeError("Outlier detection left too few card images to cluster.")

    cv2.setRNGSeed(args.seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.005)
    compactness, card_labels, centers = cv2.kmeans(
        reduced[card_indices], raw_cluster_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    raw_labels = np.full(len(valid_paths), -1, dtype=np.int32)
    raw_labels[card_indices] = card_labels.flatten()
    labels = raw_labels.copy()
    prototype_confidence = np.ones(len(valid_paths), dtype=np.float32)
    if canonical_output:
        labels, prototype_confidence = prototype_labels(
            images, valid_paths, card_indices
        )
        low_colour = np.array(
            [
                np.ptp(
                    cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)[18:101, 12:84]
                        .astype(np.float32),
                    axis=2,
                ).mean() < 12.0
                for image in images
            ]
        )
        labels[low_colour] = -1
        labels[empty] = 3
        labels = smooth_single_frame_flickers(labels, valid_paths)
    final_unknown = labels < 0

    # Distance to assigned centre is useful for auditing cluster edges.
    center_distance = np.full(len(valid_paths), np.nan, dtype=np.float32)
    center_distance[card_indices] = np.linalg.norm(
        reduced[card_indices] - centers[raw_labels[card_indices]], axis=1
    )

    assignments_path = args.output / "assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path", "slot", "filename", "cluster_id",
                "is_unknown", "unknown_reason", "cluster_distance", "outlier_score",
            ],
        )
        writer.writeheader()
        for index, path in enumerate(valid_paths):
            reason = (
                "empty" if empty[index] else
                ("transition" if transition[index] else
                 ("visual_outlier" if isolated[index] else
                  ("ambiguous_card" if labels[index] < 0 else "")))
            )
            writer.writerow(
                {
                    "image_path": str(path.resolve()),
                    "slot": path.parent.name,
                    "filename": path.name,
                    "cluster_id": int(labels[index]),
                    "is_unknown": bool(final_unknown[index]),
                    "unknown_reason": reason,
                    "cluster_distance": "" if unknown[index] else f"{center_distance[index]:.6f}",
                    "outlier_score": f"{outlier_score[index]:.6f}",
                }
            )

    cluster_summary: dict[str, dict[str, object]] = {}
    overview: list[tuple[Path, str]] = []
    output_cluster_ids = list(CANONICAL_LABELS) if canonical_output else list(
        range(args.clusters)
    )
    for cluster_id in output_cluster_ids:
        indices = np.flatnonzero(labels == cluster_id)
        ranking = np.nan_to_num(center_distance[indices], nan=np.inf)
        ordered = indices[np.argsort(ranking)]
        count = min(args.examples_per_cluster, len(indices))
        core_count = (count + 2) // 3
        diverse_count = (count - core_count + 1) // 2
        edge_count = count - core_count - diverse_count
        selected = list(ordered[:core_count])
        if diverse_count:
            quantiles = np.linspace(core_count, len(ordered) - edge_count - 1, diverse_count)
            selected.extend(ordered[np.round(quantiles).astype(int)])
        if edge_count:
            selected.extend(ordered[-edge_count:])
        examples = [
            (
                valid_paths[index],
                (
                    "core" if rank < core_count else
                    ("diverse" if rank < core_count + diverse_count else "edge")
                )
                + (
                    f" d={center_distance[index]:.2f}"
                    if np.isfinite(center_distance[index]) else ""
                ),
            )
            for rank, index in enumerate(selected)
        ]
        sheet_path = args.output / f"cluster_{cluster_id:02d}.jpg"
        contact_sheet(
            examples,
            title=(
                f"Cluster {cluster_id} {CANONICAL_LABELS.get(cluster_id, '')}"
                f" - {len(indices)} images - core/diverse/edge"
            ),
            output_path=sheet_path,
        )
        if len(indices):
            overview.append((valid_paths[ordered[0]], f"cluster {cluster_id} n={len(indices)}"))
        finite_distances = center_distance[indices][np.isfinite(center_distance[indices])]
        cluster_summary[str(cluster_id)] = {
            "count": int(len(indices)),
            "contact_sheet": str(sheet_path),
            "median_center_distance": (
                float(np.median(finite_distances)) if len(finite_distances) else None
            ),
            "suggested_label": CANONICAL_LABELS.get(cluster_id),
        }

    unknown_indices = np.flatnonzero(final_unknown)
    unknown_order = unknown_indices[np.argsort(outlier_score[unknown_indices])[::-1]]
    unknown_selected = unknown_order[: args.examples_per_cluster]
    contact_sheet(
        [
            (
                valid_paths[index],
                (
                    "empty" if empty[index] else
                    ("transition" if transition[index] else "outlier")
                ) + f" s={outlier_score[index]:.2f}",
            )
            for index in unknown_selected
        ],
        title=f"Unknown - {len(unknown_indices)} empty/loading/transition images",
        output_path=args.output / "cluster_unknown.jpg",
    )
    contact_sheet(
        overview, "Cluster overview - nearest image to each centre",
        args.output / "cluster_overview.jpg", columns=5,
    )

    summary = {
        "source_slots_dir": str(args.slots_dir),
        "image_count": len(valid_paths),
        "cluster_count": len(output_cluster_ids),
        "raw_cluster_count": raw_cluster_count,
        "unknown_count": int(final_unknown.sum()),
        "empty_count": int(empty.sum()),
        "transition_count": int((transition & ~empty).sum()),
                "visual_outlier_count": int((isolated & ~empty & ~transition).sum()),
        "ambiguous_card_count": int(((labels < 0) & ~unknown).sum()),
        "compactness": float(compactness),
        "clusters": cluster_summary,
        "unknown": {
            "count": int(final_unknown.sum()),
            "cluster_id": -1,
            "contact_sheet": str(args.output / "cluster_unknown.jpg"),
        },
    }
    summary_path = args.output / "cluster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_path = args.output / "clustering_model.npz"
    np.savez_compressed(
        model_path,
        centers=centers,
        feature_center=feature_center,
        feature_scale=feature_scale,
        projection=projection,
        reduced_center=reduced_center,
        reduced_scale=reduced_scale,
    )
    print(f"Assignments:  {assignments_path}")
    print(f"Summary:      {summary_path}")
    print(f"Model:        {model_path}")
    if canonical_output:
        print(f"Empty:        {int(empty.sum())} images (cluster_id=3)")
        print(f"Unknown:      {int(final_unknown.sum())} images (cluster_id=-1)")
    else:
        print(f"Unknown:      {int(unknown.sum())} images (cluster_id=-1)")
    print(f"Contact sheets saved in: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
