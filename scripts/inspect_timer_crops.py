from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


TIMESTAMP_RE=re.compile(r"_t(\d+)ms")


def timestamp_ms(path:Path)->int:
    match=TIMESTAMP_RE.search(path.name)
    if not match:raise ValueError(path.name)
    return int(match.group(1))


def parse_args():
    parser=argparse.ArgumentParser(description="Inspect fixed-layout Clash Royale timer crops.")
    parser.add_argument("--crops-root",type=Path,default=Path("data/crops"))
    parser.add_argument("--matches",nargs="+",default=[f"match_{i:03d}" for i in range(1,10)])
    parser.add_argument("--output",type=Path,default=Path("data/timer_analysis"))
    parser.add_argument("--examples-per-match",type=int,default=3)
    return parser.parse_args()


def features(image):
    resized=cv2.resize(image,(125,83),interpolation=cv2.INTER_AREA);hsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    digit=gray[29:82,18:113];white=(digit>180).astype(np.uint8)*255
    contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    points=[cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour)>=3]
    if points:
        x1=min(x for x,y,w,h in points)+18;y1=min(y for x,y,w,h in points)+29;x2=max(x+w for x,y,w,h in points)+18;y2=max(y+h for x,y,w,h in points)+29;bbox=[x1,y1,x2,y2]
    else:bbox=None
    return {"brightness":float(hsv[...,2].mean()),"contrast":float(gray.std()),"white_fraction":float((white>0).mean()),"digit_bbox":bbox}


def write_sheet(records,path,title):
    columns,width,height,header=6,155,145,42;canvas=np.full((header+max(1,int(np.ceil(len(records)/columns)))*height,columns*width,3),25,np.uint8)
    cv2.putText(canvas,title,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),2)
    for i,row in enumerate(records):
        image=cv2.imread(row["image_path"]);image=cv2.resize(image,(125,83));r,c=divmod(i,columns);x=c*width+15;y=header+r*height
        canvas[y:y+83,x:x+125]=image;cv2.putText(canvas,f"{row['match_id']} {row['timestamp_ms']/1000:.1f}s",(c*width+3,y+104),cv2.FONT_HERSHEY_SIMPLEX,.36,(210,230,255),1)
        cv2.putText(canvas,f"B={row['brightness']:.0f}",(c*width+3,y+122),cv2.FONT_HERSHEY_SIMPLEX,.34,(200,200,200),1)
    if not cv2.imwrite(str(path),canvas):raise RuntimeError(path)


def closest(paths,target,count):
    return sorted(paths,key=lambda path:abs(timestamp_ms(path)-target))[:count]


def main():
    args=parse_args();args.output.mkdir(parents=True,exist_ok=True);all_rows=[];by_match={};dimensions=set();transition=[]
    for match in args.matches:
        paths=sorted((args.crops_root/match/"timer").glob("*.jpg"),key=timestamp_ms)
        if not paths:raise FileNotFoundError(f"No timer crops for {match}")
        rows=[];previous=None
        for path in paths:
            image=cv2.imread(str(path));dimensions.add((image.shape[1],image.shape[0]));row={"image_path":str(path.resolve()),"match_id":match,"timestamp_ms":timestamp_ms(path),**features(image)}
            gray=cv2.cvtColor(cv2.resize(image,(125,83)),cv2.COLOR_BGR2GRAY);row["adjacent_difference"]=float(np.mean(cv2.absdiff(previous,gray))) if previous is not None else 0.0;previous=gray
            rows.append(row);all_rows.append(row)
        by_match[match]=(paths,rows);transition.extend(sorted(rows,key=lambda row:row["adjacent_difference"],reverse=True)[:args.examples_per_match])
    categories={"early_match":20000,"around_2_00":60000,"around_1_00":120000,"final_30_seconds":150000,"overtime":210000}
    outputs={}
    for name,target in categories.items():
        records=[]
        for match,(paths,rows) in by_match.items():
            lookup={row["image_path"]:row for row in rows}
            records.extend(lookup[str(path.resolve())] for path in closest(paths,target,args.examples_per_match))
        filename=f"{name}.jpg";write_sheet(records,args.output/filename,name.replace("_"," "));outputs[name]=filename
    transition=sorted(transition,key=lambda row:row["adjacent_difference"],reverse=True);write_sheet(transition,args.output/"timer_transitions.jpg","largest adjacent timer-crop changes");outputs["transitions"]="timer_transitions.jpg"
    bboxes=[row["digit_bbox"] for row in all_rows if row["digit_bbox"]]
    bbox_array=np.asarray(bboxes)
    payload={"matches":args.matches,"crop_counts":{m:len(rows) for m,(paths,rows) in by_match.items()},"dimensions":[list(v) for v in sorted(dimensions)],
             "brightness":{"minimum":min(r["brightness"] for r in all_rows),"median":float(np.median([r["brightness"] for r in all_rows])),"maximum":max(r["brightness"] for r in all_rows)},
             "digit_layout":{"canonical_crop_size":[125,83],"median_detected_bbox":np.median(bbox_array,axis=0).round(1).tolist(),"bbox_standard_deviation":bbox_array.std(axis=0).round(2).tolist(),
                 "assessment":"Digits and colon use the same lower fixed layout in regulation and overtime; only the header/background changes."},
             "layouts":["regulation header: Time left","overtime header: Overtime","non-game/transition crop"],
             "recommended_method":"Fixed-position segmentation of M:SS followed by per-position digit template matching. Detect regulation/overtime header separately, reject non-game and transition crops, then apply countdown physics smoothing. General OCR is unnecessary.",
             "outputs":outputs}
    (args.output/"timer_crop_analysis.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
