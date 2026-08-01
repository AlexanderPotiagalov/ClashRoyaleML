import numpy as np

from scripts.elixir_utils import TemporalElixirSmoother, UNKNOWN, predict_feature, train_prototype_model
from scripts.train_elixir_classifier import choose_match_split


def synthetic_model():
    samples=[]
    for value in range(11):
        samples.extend([(np.asarray([value/10, value/20],np.float32),value) for _ in range(3)])
    return train_prototype_model(samples)


def test_values_zero_and_ten():
    model=synthetic_model()
    model["unknown_threshold"]=0.0
    assert predict_feature(np.asarray([0,0],np.float32),model)[0]==0
    assert predict_feature(np.asarray([1,.5],np.float32),model)[0]==10


def test_unknown_rejection():
    model=synthetic_model()
    assert predict_feature(np.asarray([50,50],np.float32),model,threshold=.5)[0]==UNKNOWN


def test_temporal_smoothing_blocks_impossible_jumps_and_unexplained_spend():
    smoother=TemporalElixirSmoother()
    assert smoother.update(5,.9,0)==5
    assert smoother.update(10,.9,500)==6
    assert smoother.update(2,.9,1000)==6
    assert smoother.update(2,.9,1500,card_play_cost=4)==2


def test_match_level_elixir_split():
    split=choose_match_split(["match_001","match_002","match_003","match_004"])
    flattened=split["train_matches"]+split["validation_matches"]+split["test_matches"]
    assert len(flattened)==len(set(flattened))
