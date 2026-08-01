import numpy as np

from scripts.timer_utils import UNKNOWN,TimerSmoother,format_timer,parse_timer_text,phase_for,predict_timer_image
from scripts.train_timer_recognizer import choose_match_split


def test_required_timer_values():
    assert parse_timer_text("2:59")==179
    assert parse_timer_text("1:00")==60
    assert parse_timer_text("0:09")==9
    assert parse_timer_text("0:00")==0
    assert format_timer(179)=="2:59"


def test_overtime_value_and_phase_are_separate():
    rules={"overtime":[{"minimum_seconds":61,"phase":"double"},{"minimum_seconds":0,"phase":"triple"}]}
    assert parse_timer_text("3:00")==180
    assert phase_for(180,"overtime",rules)=="double"


def test_bad_crop_is_unknown():
    all_digits={str(i):[(np.ones(540,dtype=float)*i/10).tolist()] for i in range(10)}
    model={"position_exemplars":[all_digits,all_digits,all_digits],"unknown_threshold":.2}
    text,seconds,confidence,_=predict_timer_image(np.zeros((83,125,3),np.uint8),model)
    assert text==UNKNOWN and seconds is None


def test_impossible_temporal_jump_and_overtime_transition():
    smoother=TimerSmoother();assert smoother.update(102,0,"regulation")==102
    assert smoother.update(138,500,"regulation")<=102
    smoother=TimerSmoother();assert smoother.update(1,0,"regulation")==1
    assert smoother.update(180,1000,"overtime")==180


def test_timer_match_split_has_no_leakage():
    split=choose_match_split(["match_001","match_008","match_009"])
    values=split["train_matches"]+split["validation_matches"]+split["test_matches"]
    assert len(values)==len(set(values))
