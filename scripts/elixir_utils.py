from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


TIMESTAMP_RE = re.compile(r"_t(\d+)ms")
UNKNOWN = -1


def timestamp_ms(path: str | Path) -> int:
    match = TIMESTAMP_RE.search(Path(path).name)
    if not match: raise ValueError(f"Cannot parse timestamp: {path}")
    return int(match.group(1))


def extract_features(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (554, 61), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    number = gray[0:29, 40:96]
    number = cv2.resize(number, (28, 16), interpolation=cv2.INTER_AREA)
    number = np.clip((number.astype(np.float32) - 80) / 175, 0, 1)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV); bar = hsv[14:57, 90:548]
    purple = ((bar[...,0]>=130)&(bar[...,0]<=179)&(bar[...,1]>=65)&(bar[...,2]>=55))
    segments=[]
    for index in range(10):
        left=round(index*bar.shape[1]/10); right=round((index+1)*bar.shape[1]/10)
        segments.append(float(purple[:,left:right].mean()))
    fill=float(purple.mean())
    return np.concatenate([number.ravel()*1.8, np.asarray(segments,dtype=np.float32)*4, [fill*4]]).astype(np.float32)


def train_prototype_model(samples: list[tuple[np.ndarray, int]]) -> dict[str, object]:
    missing=[value for value in range(11) if not any(label==value for _,label in samples)]
    if missing: raise ValueError(f"Training labels lack elixir values: {missing}")
    prototypes={}; within=[]
    for value in range(11):
        matrix=np.vstack([feature for feature,label in samples if label==value])
        prototype=matrix.mean(0); prototypes[str(value)]=prototype.tolist()
        within.extend(np.linalg.norm(matrix-prototype,axis=1).tolist())
    scale=max(float(np.percentile(within,90)),1e-4)
    return {"method":"deterministic_template_segments_v1","feature_size":len(samples[0][0]),
            "prototypes":prototypes,"distance_scale":scale,"unknown_threshold":0.55}


def predict_feature(feature: np.ndarray, model: dict[str, object], threshold: float | None=None):
    scored=sorted((float(np.linalg.norm(feature-np.asarray(vector,np.float32))),int(label))
                  for label,vector in model["prototypes"].items())
    best_distance,value=scored[0]; second_distance=scored[1][0]
    distance_score=float(np.exp(-best_distance/max(float(model["distance_scale"]),1e-6)))
    margin_score=float(np.clip((second_distance-best_distance)/max(second_distance,1e-6),0,1))
    confidence=float(np.sqrt(distance_score*margin_score))
    cutoff=float(model.get("unknown_threshold",.55) if threshold is None else threshold)
    return (value if confidence>=cutoff else UNKNOWN), confidence, best_distance


def load_model(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class TemporalElixirSmoother:
    def __init__(self): self.value=None; self.timestamp=None

    def update(self, value: int, confidence: float, timestamp: int, card_play_cost: int | None=None) -> int:
        if value==UNKNOWN or confidence<=0: return self.value if self.value is not None else UNKNOWN
        if self.value is None: self.value,self.timestamp=value,timestamp; return value
        elapsed=max(0,timestamp-(self.timestamp or timestamp)); maximum_increase=max(1,int(np.ceil(elapsed/1000)))
        if value>self.value+maximum_increase: value=self.value+maximum_increase
        if value<self.value:
            if card_play_cost is None: value=self.value
            else:
                expected=max(0,self.value-card_play_cost)
                if abs(value-expected)>1: value=expected
        self.value=int(np.clip(value,0,10)); self.timestamp=timestamp; return self.value
