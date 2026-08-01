from __future__ import annotations

import argparse,csv,json
from pathlib import Path
import cv2,numpy as np

try:
    from timer_utils import UNKNOWN,predict_timer_image,train_template_model
except ModuleNotFoundError:
    from scripts.timer_utils import UNKNOWN,predict_timer_image,train_template_model


def parse_args():
    parser=argparse.ArgumentParser(description="Calibrate deterministic fixed-position timer templates.")
    parser.add_argument("--labels",type=Path,default=Path("data/timer_labels/timer_labels.csv"));parser.add_argument("--output",type=Path,default=Path("models/timer_recognizer_v1/model.json"))
    parser.add_argument("--split-config",type=Path,default=Path("config/timer_training_split.json"));parser.add_argument("--validation-match");parser.add_argument("--test-match")
    return parser.parse_args()


def choose_match_split(matches,validation=None,test=None):
    matches=sorted(set(matches))
    if len(matches)<3:raise ValueError("Need sparse labels from three matches. Run label_timer_frames.py --stratified")
    validation=validation or matches[-2];test=test or matches[-1];train=[m for m in matches if m not in {validation,test}]
    flattened=train+[validation,test]
    if len(flattened)!=len(set(flattened)):raise ValueError("Match-level split leakage")
    return {"train_matches":train,"validation_matches":[validation],"test_matches":[test]}


def main():
    args=parse_args()
    if not args.labels.is_file():raise FileNotFoundError("Run: python scripts/label_timer_frames.py --stratified")
    with args.labels.open(newline="",encoding="utf-8") as handle:rows=[r for r in csv.DictReader(handle) if r["timer_text"]!=UNKNOWN]
    split=choose_match_split([r["match_id"] for r in rows],args.validation_match,args.test_match);train=[r for r in rows if r["match_id"] in split["train_matches"]];validation=[r for r in rows if r["match_id"] in split["validation_matches"]]
    samples=[]
    for row in train:
        image=cv2.imread(row["image_path"]);samples.append((image,row["timer_text"]))
    model=train_template_model(samples);best=(0.0,.5)
    for threshold in np.linspace(.15,.75,25):
        correct=0
        for row in validation:
            text,_,_,_=predict_timer_image(cv2.imread(row["image_path"]),model,float(threshold));correct+=text==row["timer_text"]
        accuracy=correct/max(1,len(validation))
        if accuracy>best[0]:best=(accuracy,float(threshold))
    final_samples=list(samples)
    for row in validation:
        final_samples.append((cv2.imread(row["image_path"]),row["timer_text"]))
    model=train_template_model(final_samples)
    model.update({"unknown_threshold":best[1],"validation_accuracy_before_refit":best[0],
                  "calibration_matches":split["train_matches"]+split["validation_matches"],"split":split})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(model,indent=2),encoding="utf-8");args.split_config.parent.mkdir(parents=True,exist_ok=True);args.split_config.write_text(json.dumps(split,indent=2),encoding="utf-8")
    print(json.dumps({"method":model["method"],"split":split,"validation_accuracy_before_refit":best[0],"unknown_threshold":best[1],"final_template_matches":model["calibration_matches"]},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
