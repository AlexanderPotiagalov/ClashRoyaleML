from scripts.recover_card_labels import recover_rows


LABELS = {0: "barbarian_barrel", 3: "empty", 5: "cannon"}


def row(timestamp: int, cluster_id: int, reason: str = "") -> dict[str, str]:
    return {
        "slot": "slot_1",
        "filename": f"frame_000000_t{timestamp:09d}ms.jpg",
        "timestamp_ms": str(timestamp),
        "cluster_id": str(cluster_id),
        "unknown_reason": reason,
        "is_unknown": str(cluster_id < 0),
    }


def test_matching_temporal_anchors_recover_unknown() -> None:
    rows = [row(0, 5), row(500, -1, "ambiguous_card"), row(1000, 5)]
    recovered = recover_rows(rows, LABELS, max_gap_ms=1000)
    assert recovered[1]["cluster_id"] == "5"
    assert recovered[1]["label"] == "cannon"
    assert recovered[1]["recovered"] == "True"


def test_different_anchors_do_not_recover() -> None:
    rows = [row(0, 5), row(500, -1, "ambiguous_card"), row(1000, 0)]
    recovered = recover_rows(rows, LABELS, max_gap_ms=1000)
    assert recovered[1]["cluster_id"] == "-1"


def test_empty_and_transition_are_hard_barriers() -> None:
    empty_rows = [
        row(0, 5), row(500, 3), row(1000, -1, "ambiguous_card"), row(1500, 5)
    ]
    transition_rows = [
        row(0, 5), row(500, -1, "transition"),
        row(1000, -1, "ambiguous_card"), row(1500, 5),
    ]
    assert recover_rows(empty_rows, LABELS, 2000)[2]["cluster_id"] == "-1"
    assert recover_rows(transition_rows, LABELS, 2000)[2]["cluster_id"] == "-1"


def test_gap_limit_is_applied_to_both_anchors() -> None:
    rows = [row(0, 5), row(2500, -1, "ambiguous_card"), row(3000, 5)]
    recovered = recover_rows(rows, LABELS, max_gap_ms=1000)
    assert recovered[1]["cluster_id"] == "-1"
