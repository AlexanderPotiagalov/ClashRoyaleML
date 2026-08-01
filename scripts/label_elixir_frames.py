from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    from elixir_utils import extract_features,predict_feature,timestamp_ms,train_prototype_model
except ModuleNotFoundError:
    from scripts.elixir_utils import extract_features,predict_feature,timestamp_ms,train_prototype_model


def parse_args():
    parser=argparse.ArgumentParser(description="Keyboard-label elixir crops (0-9, T=10, U=unknown, Q=save/quit).")
    parser.add_argument("--crops-root",type=Path,default=Path("data/crops"))
    parser.add_argument("--matches",nargs="+",default=[f"match_{i:03d}" for i in range(1,10)])
    parser.add_argument("--output",type=Path,default=Path("data/elixir_labels/elixir_labels.csv"))
    parser.add_argument("--identical-threshold",type=float,default=1.5)
    parser.add_argument("--stratified",action="store_true",help="Review a small value-balanced subset instead of walking every frame")
    parser.add_argument("--samples-per-value",type=int,default=2,help="Per match and predicted value in stratified mode")
    return parser.parse_args()


def load_existing(path):
    if not path.exists(): return {}
    with path.open(newline="",encoding="utf-8") as handle: return {row["image_path"]:row for row in csv.DictReader(handle)}


def save(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True); fields=["image_path","match_id","timestamp_ms","elixir","source","propagated"]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(sorted(rows.values(),key=lambda r:(r["match_id"],int(r["timestamp_ms"]))))


def stratified_paths(paths,rows,samples_per_value):
    labelled=[row for row in rows.values() if 0<=int(row["elixir"])<=10]
    values={int(row["elixir"]) for row in labelled}
    if values!=set(range(11)):
        raise ValueError(f"Stratified assistance needs existing examples of 0-10; missing {sorted(set(range(11))-values)}")
    samples=[]
    for row in labelled:
        image=cv2.imread(row["image_path"]); samples.append((extract_features(image),int(row["elixir"])))
    model=train_prototype_model(samples); chosen=[]
    for match in sorted({match for match,_ in paths}):
        candidates=[]
        for candidate_match,path in paths:
            resolved=str(path.resolve())
            if candidate_match!=match or resolved in rows: continue
            image=cv2.imread(str(path)); prediction,confidence,_=predict_feature(extract_features(image),model,threshold=0.0)
            candidates.append((confidence,prediction,path))
        for value in range(11):
            selected=[]
            for confidence,_,path in sorted((row for row in candidates if row[1]==value),reverse=True,key=lambda row:row[0]):
                if all(abs(timestamp_ms(path)-timestamp_ms(existing))>=5000 for existing in selected):
                    selected.append(path); chosen.append((match,path,value,confidence))
                if len(selected)>=samples_per_value: break
    return chosen


def main():
    args=parse_args(); paths=[]
    for match in args.matches:
        paths.extend((match,path) for path in sorted((args.crops_root/match/"elixir").glob("*.jpg"),key=timestamp_ms))
    rows=load_existing(args.output)
    assisted=stratified_paths(paths,rows,args.samples_per_value) if args.stratified else None
    work=[(match,path,suggestion,confidence) for match,path,suggestion,confidence in assisted] if assisted is not None else [(match,path,None,None) for match,path in paths if str(path.resolve()) not in rows]
    position=0
    window="Elixir labelling: 0-9, T=10, U=unknown, Q=save/quit"
    try:
        while position<len(work):
            match,path,suggestion,suggestion_confidence=work[position]; image=cv2.imread(str(path))
            display=cv2.resize(image,None,fx=1.4,fy=1.4,interpolation=cv2.INTER_NEAREST)
            detail=f"{match} {position+1}/{len(work)}"
            if suggestion is not None: detail+=f" suggested={suggestion} ({suggestion_confidence:.2f})"
            cv2.putText(display,detail,(5,display.shape[0]-5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            cv2.imshow(window,display); key=cv2.waitKey(0)&0xFF
            if ord('0')<=key<=ord('9'): value=key-ord('0')
            elif key in (ord('t'),ord('T')): value=10
            elif key in (ord('u'),ord('U')): value=-1
            elif key in (ord('q'),ord('Q'),27): break
            else: continue
            resolved=str(path.resolve()); rows[resolved]={"image_path":resolved,"match_id":match,"timestamp_ms":timestamp_ms(path),"elixir":value,"source":"manual","propagated":"false"}
            position+=1
            if not args.stratified:
                previous=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                while position<len(work) and work[position][0]==match:
                    next_path=work[position][1]; next_image=cv2.imread(str(next_path)); current=cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
                    if float(np.mean(cv2.absdiff(previous,current)))>args.identical_threshold: break
                    resolved=str(next_path.resolve()); rows[resolved]={"image_path":resolved,"match_id":match,"timestamp_ms":timestamp_ms(next_path),"elixir":value,"source":"visual_identity","propagated":"true"}
                    previous=current; position+=1
            if len(rows)%25==0: save(rows,args.output)
    finally:
        save(rows,args.output); cv2.destroyAllWindows()
    print(f"Saved {len(rows)} labels to {args.output}")
    if args.stratified: print(f"Stratified review decisions this run: {len(work)}")
    return 0


if __name__=="__main__": raise SystemExit(main())
