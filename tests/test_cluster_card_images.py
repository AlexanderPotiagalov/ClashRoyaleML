import numpy as np
from pathlib import Path

from scripts.cluster_card_images import (
    card_feature,
    empty_slot_mask,
    smooth_single_frame_flickers,
    transition_frame_mask,
)


def test_card_feature_is_finite_and_deterministic() -> None:
    image = np.random.default_rng(4).integers(0, 256, (188, 133, 3), dtype=np.uint8)
    first = card_feature(image)
    second = card_feature(image)
    assert first.ndim == 1
    assert first.size > 1000
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_blue_placeholder_is_empty() -> None:
    image = np.full((188, 133, 3), (210, 120, 20), dtype=np.uint8)
    assert empty_slot_mask([image]).tolist() == [True]


def test_white_chat_overlay_is_transition() -> None:
    image = np.zeros((188, 133, 3), dtype=np.uint8)
    image[35:160] = 255
    assert transition_frame_mask([image]).tolist() == [True]


def test_single_frame_class_flicker_is_smoothed() -> None:
    paths = [Path("slot_1") / f"frame_{index:03d}.jpg" for index in range(5)]
    labels = np.array([2, 2, 7, 2, 2], dtype=np.int32)
    assert smooth_single_frame_flickers(labels, paths).tolist() == [2, 2, -1, 2, 2]
