from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from elixir_utils import extract_features, predict_feature, train_prototype_model
except ModuleNotFoundError:
    from scripts.elixir_utils import extract_features, predict_feature, train_prototype_model


def parse_args():
    parser=argparse.ArgumentParser(description="Train the deterministic elixir numeral/segment baseline.")
    parser.add_argument("--labels",type=Path,default=Path("data/elixir_labels/elixir_labels.csv"))
    parser.add_argument("--output",type=Path,default=Path("models/elixir_classifier_v1/model.json"))
    parser.add_argument("--split-config",type=Path,default=Path("config/elixir_training_split.json"))
    parser.add_argument("--validation-match"); parser.add_argument("--test-match")
    return parser.parse_args()


def choose_match_split(matches,validation=None,test=None):
    matches=sorted(set(matches))
    if len(matches)<3:
        raise ValueError("Need labels from 3 matches, not complete matches. Run: python scripts/label_elixir_frames.py --stratified --matches match_008 match_009")
    validation=validation or matches[-2]; test=test or matches[-1]
    train=[match for match in matches if match not in {validation,test}]
    split={"train_matches":train,"validation_matches":[validation],"test_matches":[test]}
    flattened=train+[validation,test]
    if len(flattened)!=len(set(flattened)): raise ValueError("A match appears in multiple splits")
    return split


def load_rows(path):
    if not path.is_file(): raise FileNotFoundError(f"Run label_elixir_frames.py first: {path}")
    with path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    return [row for row in rows if 0<=int(row["elixir"])<=10]


def main():
    args=parse_args(); rows=load_rows(args.labels); split=choose_match_split([row["match_id"] for row in rows],args.validation_match,args.test_match)
    train_rows=[row for row in rows if row["match_id"] in split["train_matches"]]
    counts=Counter(int(row["elixir"]) for row in train_rows); missing=[v for v in range(11) if not counts[v]]
    if missing: raise ValueError(f"Training matches lack elixir labels: {missing}")
    cache={}
    def feature(row):
        path=row["image_path"]
        if path not in cache:
            image=cv2.imread(path)
            if image is None: raise FileNotFoundError(path)
            cache[path]=extract_features(image)
        return cache[path]
    model=train_prototype_model([(feature(row),int(row["elixir"])) for row in train_rows])
    validation_rows=[row for row in rows if row["match_id"] in split["validation_matches"]]
    best=(0.0,.55)
    for threshold in np.linspace(.25,.80,23):
        predictions=[predict_feature(feature(row),model,float(threshold))[0] for row in validation_rows]
        accuracy=sum(pred==int(row["elixir"]) for pred,row in zip(predictions,validation_rows))/max(1,len(validation_rows))
        if accuracy>best[0]: best=(accuracy,float(threshold))
    model.update({"unknown_threshold":best[1],"validation_accuracy":best[0],"split":split,
                  "training_class_counts":{str(v):counts[v] for v in range(11)}})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(model,indent=2),encoding="utf-8")
    args.split_config.parent.mkdir(parents=True,exist_ok=True); args.split_config.write_text(json.dumps(split,indent=2),encoding="utf-8")
    print(json.dumps({"method":model["method"],"split":split,"validation_accuracy":best[0],"unknown_threshold":best[1]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
