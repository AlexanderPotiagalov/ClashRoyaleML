from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

try:
    from elixir_utils import TemporalElixirSmoother,UNKNOWN,extract_features,load_model,predict_feature,timestamp_ms
except ModuleNotFoundError:
    from scripts.elixir_utils import TemporalElixirSmoother,UNKNOWN,extract_features,load_model,predict_feature,timestamp_ms


def parse_args():
    parser=argparse.ArgumentParser(description="Predict elixir from one crop or a chronological directory.")
    parser.add_argument("input",type=Path);parser.add_argument("--model",type=Path,default=Path("models/elixir_classifier_v1/model.json"))
    parser.add_argument("--confidence-threshold",type=float);parser.add_argument("--temporal-smoothing",action="store_true")
    parser.add_argument("--card-plays-csv",type=Path,help="Optional CSV with timestamp_ms and elixir_cost")
    return parser.parse_args()


def main():
    args=parse_args(); model=load_model(args.model);paths=[args.input] if args.input.is_file() else sorted(args.input.glob("*.jpg"),key=timestamp_ms)
    plays={}
    if args.card_plays_csv:
        with args.card_plays_csv.open(newline="",encoding="utf-8") as handle: plays={int(r["timestamp_ms"]):int(r["elixir_cost"]) for r in csv.DictReader(handle)}
    smoother=TemporalElixirSmoother();results=[]
    for index,path in enumerate(paths):
        image=cv2.imread(str(path));value,confidence,_=predict_feature(extract_features(image),model,args.confidence_threshold)
        stamp=timestamp_ms(path) if "_t" in path.name else index*500; raw=value
        if args.temporal_smoothing: value=smoother.update(value,confidence,stamp,plays.get(stamp))
        results.append({"image_path":str(path.resolve()),"elixir":None if value==UNKNOWN else value,"confidence":confidence,"method":model["method"],"raw_elixir":None if raw==UNKNOWN else raw})
    print(json.dumps(results,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
