from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


TIMESTAMP_RE = re.compile(r"_t(\d+)ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect visual signals in multi-match elixir crops.")
    parser.add_argument("--crops-root", type=Path, default=Path("data/crops"))
    parser.add_argument("--matches", nargs="+", default=[f"match_{i:03d}" for i in range(1, 10)])
    parser.add_argument("--output", type=Path, default=Path("data/elixir_analysis"))
    parser.add_argument("--samples-per-match", type=int, default=48)
    return parser.parse_args()


def timestamp_ms(path: Path) -> int:
    match = TIMESTAMP_RE.search(path.name)
    if not match: raise ValueError(f"Missing timestamp: {path.name}")
    return int(match.group(1))


def elixir_features(image: np.ndarray) -> dict[str, object]:
    resized = cv2.resize(image, (554, 61), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    bar = hsv[14:57, 90:548]
    purple = ((bar[..., 0] >= 130) & (bar[..., 0] <= 179) &
              (bar[..., 1] >= 65) & (bar[..., 2] >= 55))
    segment_ratios = []
    for index in range(10):
        left = round(index * bar.shape[1] / 10); right = round((index + 1) * bar.shape[1] / 10)
        segment_ratios.append(float(purple[:, left:right].mean()))
    filled_estimate = int(sum(value >= .16 for value in segment_ratios))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    number = gray[0:34, 43:103]
    white_fraction = float((number > 205).mean())
    return {
        "brightness": float(hsv[..., 2].mean()),
        "purple_fill_ratio": float(purple.mean()),
        "segment_purple_ratios": segment_ratios,
        "filled_segment_estimate": filled_estimate,
        "number_white_fraction": white_fraction,
    }


def pattern_feature(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (96, 16), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    purple = (((hsv[..., 0] >= 130) & (hsv[..., 0] <= 179) & (hsv[..., 1] >= 65)) * 255).astype(np.uint8)
    return cv2.resize(purple, (24, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255


def write_sheet(records, output, title, limit=72):
    selected = records[:limit]; columns, width, height, header = 3, 590, 115, 42
    sheet = np.full((header + max(1, int(np.ceil(len(selected)/columns))) * height,
                     columns*width, 3), 25, np.uint8)
    cv2.putText(sheet, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    for index, record in enumerate(selected):
        image = cv2.imread(record["image_path"])
        if image is None: continue
        image = cv2.resize(image, (554, 61), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns); x=column*width+8; y=header+row*height
        sheet[y:y+61,x:x+554]=image
        text=(f"{record['match_id']} {record['timestamp_ms']}ms "
              f"purple={record['purple_fill_ratio']:.3f} seg={record['filled_segment_estimate']}")
        cv2.putText(sheet,text,(x,y+83),cv2.FONT_HERSHEY_SIMPLEX,.4,(210,230,255),1)
    if not cv2.imwrite(str(output),sheet): raise RuntimeError(f"Could not write {output}")


def main() -> int:
    args=parse_args()
    if args.samples_per_match < 1: raise ValueError("--samples-per-match must be positive")
    args.output.mkdir(parents=True,exist_ok=True); records=[]; pattern_rows=[]
    counts={}
    for match in args.matches:
        paths=sorted((args.crops_root/match/"elixir").glob("*.jpg"),key=timestamp_ms)
        if not paths: raise FileNotFoundError(f"No elixir crops for {match}")
        counts[match]=len(paths); positions=np.linspace(0,len(paths)-1,min(len(paths),args.samples_per_match)).astype(int)
        for position in positions:
            path=paths[position]; image=cv2.imread(str(path))
            if image is None: raise RuntimeError(f"Could not read {path}")
            row={"image_path":str(path.resolve()),"match_id":match,"timestamp_ms":timestamp_ms(path),**elixir_features(image)}
            records.append(row); pattern_rows.append(pattern_feature(image))
    features=np.asarray(pattern_rows,np.float32); cluster_count=min(12,len(records))
    _,labels,_=cv2.kmeans(features,cluster_count,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,.01),5,cv2.KMEANS_PP_CENTERS)
    for row,label in zip(records,labels.ravel()): row["visual_pattern_cluster"]=int(label)
    write_sheet(sorted(records,key=lambda r:r["brightness"]),args.output/"by_brightness.jpg","Elixir crops sorted by brightness")
    write_sheet(sorted(records,key=lambda r:r["purple_fill_ratio"]),args.output/"by_purple_fill.jpg","Elixir crops sorted by purple fill")
    write_sheet(sorted(records,key=lambda r:(r["visual_pattern_cluster"],r["purple_fill_ratio"])),args.output/"by_visual_pattern.jpg","Elixir crops grouped by visual pattern")
    estimates=Counter(int(row["filled_segment_estimate"]) for row in records)
    payload={
        "matches":args.matches,"crop_counts":counts,"sample_count":len(records),"canonical_size":[554,61],
        "signals":{
            "displayed_number":{"visible":True,"assessment":"Best direct integer signal at a fixed location; use learned templates rather than general-purpose OCR."},
            "filled_segments":{"visible":True,"assessment":"Most stable validation signal, but recharge animation can partially fill the next segment."},
            "purple_fill_ratio":{"visible":True,"assessment":"Strong continuous supporting signal; vulnerable to glow and partial segment animation."},
        },
        "recommended_method":"Use fixed-position numeral template recognition as the primary integer signal; validate it with per-segment purple occupancy and total fill for confidence/rejection.",
        "filled_segment_estimate_distribution":dict(sorted(estimates.items())),
        "outputs":{"brightness":"by_brightness.jpg","purple_fill":"by_purple_fill.jpg","visual_pattern":"by_visual_pattern.jpg"},
    }
    (args.output/"elixir_crop_analysis.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
