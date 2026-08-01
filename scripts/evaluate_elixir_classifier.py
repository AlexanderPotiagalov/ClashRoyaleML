from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path("models/.cache/matplotlib").resolve()))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

try:
    from elixir_utils import UNKNOWN,extract_features,load_model,predict_feature
except ModuleNotFoundError:
    from scripts.elixir_utils import UNKNOWN,extract_features,load_model,predict_feature


def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate elixir recognition on held-out complete matches.")
    parser.add_argument("--labels",type=Path,default=Path("data/elixir_labels/elixir_labels.csv"))
    parser.add_argument("--model",type=Path,default=Path("models/elixir_classifier_v1/model.json"))
    parser.add_argument("--output",type=Path,default=Path("models/elixir_classifier_v1/evaluation"))
    return parser.parse_args()


def sheet(rows,path):
    rows=rows[:48]; columns,width,height=4,590,110; canvas=np.full((max(1,int(np.ceil(len(rows)/columns)))*height,columns*width,3),25,np.uint8)
    for i,row in enumerate(rows):
        image=cv2.imread(row["image_path"]); image=cv2.resize(image,(554,61)); r,c=divmod(i,columns); x=c*width+5;y=r*height
        canvas[y:y+61,x:x+554]=image; cv2.putText(canvas,f"true={row['true_elixir']} pred={row['predicted_elixir']} conf={row['confidence']:.2f}",(x,y+83),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,240,255),1)
    cv2.imwrite(str(path),canvas)


def main():
    args=parse_args(); model=load_model(args.model); test=set(model["split"]["test_matches"])
    with args.labels.open(newline="",encoding="utf-8") as handle: rows=[row for row in csv.DictReader(handle) if row["match_id"] in test and 0<=int(row["elixir"])<=10]
    if not rows: raise ValueError(f"No held-out labels for {sorted(test)}")
    output=[]
    for row in rows:
        image=cv2.imread(row["image_path"]); prediction,confidence,_=predict_feature(extract_features(image),model)
        output.append({"image_path":row["image_path"],"match_id":row["match_id"],"true_elixir":int(row["elixir"]),"predicted_elixir":prediction,"confidence":confidence,"correct":prediction==int(row["elixir"])})
    true=[r["true_elixir"] for r in output]; predicted=[r["predicted_elixir"] for r in output]
    errors=defaultdict(list)
    for row in output: errors[row["true_elixir"]].append(abs(row["predicted_elixir"]-row["true_elixir"]) if row["predicted_elixir"]!=UNKNOWN else None)
    metrics={"test_matches":sorted(test),"sample_count":len(rows),"exact_accuracy":sum(r["correct"] for r in output)/len(output),
             "mean_absolute_error":float(np.mean([abs(p-t) for p,t in zip(predicted,true) if p!=UNKNOWN])) if any(p!=UNKNOWN for p in predicted) else None,
             "unknown_count":sum(p==UNKNOWN for p in predicted),
             "by_true_value":{str(v):{"count":len(errors[v]),"correct":sum(r["correct"] for r in output if r["true_elixir"]==v),
                 "mean_absolute_error":float(np.mean([e for e in errors[v] if e is not None])) if any(e is not None for e in errors[v]) else None} for v in range(11)}}
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (args.output/"test_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    labels=list(range(11))+[UNKNOWN]; matrix=confusion_matrix(true,predicted,labels=labels)
    fig,ax=plt.subplots(figsize=(10,9));ax.imshow(matrix,cmap="Blues");ax.set_xticks(range(12),[str(v) for v in range(11)]+["unknown"]);ax.set_yticks(range(12),[str(v) for v in range(11)]+["unknown"]);ax.set_xlabel("Predicted");ax.set_ylabel("True")
    for i in range(12):
        for j in range(12): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8)
    fig.tight_layout();fig.savefig(args.output/"confusion_matrix.png",dpi=160);plt.close(fig)
    sheet([row for row in output if not row["correct"]],args.output/"incorrect_predictions.jpg")
    print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
