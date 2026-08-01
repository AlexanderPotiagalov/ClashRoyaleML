from __future__ import annotations

import argparse,csv,json
from collections import Counter
from pathlib import Path
import cv2,numpy as np

try:
    from timer_utils import UNKNOWN,TimerSmoother,format_timer,predict_timer_image
except ModuleNotFoundError:
    from scripts.timer_utils import UNKNOWN,TimerSmoother,format_timer,predict_timer_image


def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate timer templates on a held-out complete-match split.")
    parser.add_argument("--labels",type=Path,default=Path("data/timer_labels/timer_labels.csv"));parser.add_argument("--model",type=Path,default=Path("models/timer_recognizer_v1/model.json"));parser.add_argument("--output",type=Path,default=Path("models/timer_recognizer_v1/evaluation"))
    return parser.parse_args()


def sheet(rows,path):
    rows=rows[:48];columns,width,height=6,155,130;canvas=np.full((max(1,int(np.ceil(len(rows)/columns)))*height,columns*width,3),25,np.uint8)
    for i,row in enumerate(rows):
        image=cv2.resize(cv2.imread(row["image_path"]),(125,83));r,c=divmod(i,columns);x=c*width+15;y=r*height;canvas[y:y+83,x:x+125]=image
        cv2.putText(canvas,f"T:{row['timer_text']} P:{row['predicted_text']}",(c*width+2,y+104),cv2.FONT_HERSHEY_SIMPLEX,.34,(220,240,255),1)
    cv2.imwrite(str(path),canvas)


def main():
    args=parse_args();model=json.loads(args.model.read_text(encoding="utf-8"));test=set(model["split"]["test_matches"])
    with args.labels.open(newline="",encoding="utf-8") as handle:rows=[r for r in csv.DictReader(handle) if r["match_id"] in test and r["timer_text"]!=UNKNOWN]
    predictions=[];digit_errors=Counter();smoothers={}
    rows.sort(key=lambda row:(row["match_id"],int(row["timestamp_ms"])))
    for row in rows:
        text,seconds,confidence,layout=predict_timer_image(cv2.imread(row["image_path"]),model);truth=int(row["seconds_remaining"])
        if text!=UNKNOWN:
            for true_char,pred_char in zip(row["timer_text"].replace(":",""),text.replace(":","")):
                if true_char!=pred_char:digit_errors[f"{true_char}->{pred_char}"]+=1
        smoother=smoothers.setdefault(row["match_id"],TimerSmoother());smoothed=smoother.update(seconds,int(row["timestamp_ms"]),layout,confidence);smoothed_text=UNKNOWN if smoothed is None else format_timer(smoothed)
        predictions.append({**row,"predicted_text":text,"predicted_seconds":"" if seconds is None else seconds,"smoothed_text":smoothed_text,"smoothed_seconds":"" if smoothed is None else smoothed,"confidence":confidence,"raw_correct":text==row["timer_text"],"correct":smoothed_text==row["timer_text"]})
    accepted=[r for r in predictions if r["smoothed_seconds"]!=""];mae=float(np.mean([abs(int(r["smoothed_seconds"])-int(r["seconds_remaining"])) for r in accepted])) if accepted else None
    metrics={"test_matches":sorted(test),"sample_count":len(rows),"raw_exact_text_accuracy":sum(r["raw_correct"] for r in predictions)/len(rows),"exact_text_accuracy":sum(r["correct"] for r in predictions)/len(rows),"exact_seconds_accuracy":sum(r["correct"] for r in predictions)/len(rows),"mean_absolute_seconds_error":mae,"unknown_rejection_rate":sum(r["predicted_text"]==UNKNOWN for r in predictions)/len(rows),"per_digit_errors":dict(digit_errors)}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/"test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8");(args.output/"error_summary.json").write_text(json.dumps({"per_digit_errors":dict(digit_errors)},indent=2),encoding="utf-8")
    with (args.output/"test_predictions.csv").open("w",newline="",encoding="utf-8") as handle:w=csv.DictWriter(handle,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    sheet([r for r in predictions if not r["correct"]],args.output/"incorrect_predictions.jpg");print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
