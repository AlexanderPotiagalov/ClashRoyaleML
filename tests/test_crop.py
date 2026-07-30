from scripts.extract_frames import validate_crop


def test_valid_crop() -> None:
    assert validate_crop((0.0, 0.1, 1.0, 0.8)) == (0.0, 0.1, 1.0, 0.8)


def test_invalid_crop() -> None:
    try:
        validate_crop((0.5, 0.1, 0.4, 0.8))
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
