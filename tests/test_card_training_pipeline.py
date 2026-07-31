from pathlib import Path

import torch

from scripts.card_model_utils import VISUAL_CLASSES, create_model, load_model_bundle, logical_fields, prediction_fields
from scripts.prepare_card_training_data import validate_split


def test_match_level_split_has_no_overlap():
    split = {"train": ["match_001", "match_002"], "validation": ["match_003"], "test": ["match_004"]}
    validate_split(split)
    flattened = [match for matches in split.values() for match in matches]
    assert len(flattened) == len(set(flattened))


def test_visual_to_logical_mapping():
    assert logical_fields("cannon_evolution") == ("cannon", True)
    assert logical_fields("cannon") == ("cannon", False)
    assert logical_fields("fireball") == ("fireball", False)


def test_model_loading_and_output_fields(tmp_path: Path):
    model = create_model(len(VISUAL_CLASSES), pretrained=False)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "classes": VISUAL_CLASSES, "epoch": 0}, checkpoint)
    loaded, classes, metadata = load_model_bundle(checkpoint)
    assert loaded(torch.zeros(1, 3, 160, 112)).shape == (1, len(VISUAL_CLASSES))
    assert classes == VISUAL_CLASSES
    assert metadata["epoch"] == 0
    assert set(prediction_fields("cannon_evolution", .9, .5)) == {
        "visual_label", "logical_card", "is_evolved", "confidence"
    }


def test_unknown_rejection_below_threshold():
    result = prediction_fields("knight", .49, .50)
    assert result["visual_label"] == "unknown"
    assert result["logical_card"] == "unknown"
    assert result["is_evolved"] is False
