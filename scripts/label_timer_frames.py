from __future__ import annotations

import argparse,csv
from pathlib import Path
import cv2,numpy as np

try:
    from timer_utils import UNKNOWN,parse_timer_text,phase_for,load_phase_rules,timer_layout
    from elixir_utils import timestamp_ms
except ModuleNotFoundError:
    from scripts.timer_utils import UNKNOWN,parse_timer_text,phase_for,load_phase_rules,timer_layout
    from scripts.elixir_utils import timestamp_ms


def parse_args():
    parser=argparse.ArgumentParser(description="Label timer text; type M:SS then Enter, U unknown, Q save/quit.")
    parser.add_argument("--crops-root",type=Path,default=Path("data/crops"));parser.add_argument("--matches",nargs="+",default=["match_001","match_008","match_009"])
    parser.add_argument("--output",type=Path,default=Path("data/timer_labels/timer_labels.csv"));parser.add_argument("--phase-rules",type=Path,default=Path("config/timer_phase_rules.json"))
    parser.add_argument("--stratified",action="store_true");parser.add_argument("--training-samples",type=int,default=20);parser.add_argument("--evaluation-samples",type=int,default=12)
    return parser.parse_args()


def load(path):
    if not path.exists():return {}
    with path.open(newline="",encoding="utf-8") as handle:return {row["image_path"]:row for row in csv.DictReader(handle)}


def save(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True);fields=["image_path","match_id","timestamp_ms","timer_text","seconds_remaining","phase","layout"]
    with path.open("w",newline="",encoding="utf-8") as handle:w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(sorted(rows.values(),key=lambda row:(row["match_id"],int(row["timestamp_ms"]))))


def select(paths,count):
    if len(paths)<=count:return paths
    start=min(8,len(paths)-1);end=max(start,len(paths)-8);return [paths[i] for i in np.linspace(start,end-1,count).astype(int)]


def main():
    args=parse_args();rules=load_phase_rules(args.phase_rules);rows=load(args.output);work=[]
    for index,match in enumerate(args.matches):
        paths=sorted((args.crops_root/match/"timer").glob("*.jpg"),key=timestamp_ms);paths=[p for p in paths if str(p.resolve()) not in rows]
        chosen=select(paths,args.training_samples if index==0 else args.evaluation_samples) if args.stratified else paths
        work.extend((match,path) for path in chosen)
    position=0;buffer="";window="Timer label: M:SS Enter | U unknown | Q quit"
    try:
        while position<len(work):
            match,path=work[position];image=cv2.imread(str(path));display=cv2.resize(image,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_NEAREST)
            cv2.putText(display,f"{match} {position+1}/{len(work)} input={buffer}",(4,display.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1);cv2.imshow(window,display);key=cv2.waitKey(0)&0xFF
            if key in (ord('q'),ord('Q'),27) and not buffer:break
            if key in (8,127):buffer=buffer[:-1];continue
            if key in (ord('u'),ord('U')) and not buffer:text=UNKNOWN;seconds=None
            elif key in (10,13):
                try:text=buffer;seconds=parse_timer_text(text)
                except ValueError:continue
            elif chr(key).isdigit() or chr(key)==':':buffer+=chr(key);continue
            else:continue
            layout=timer_layout(image);resolved=str(path.resolve());rows[resolved]={"image_path":resolved,"match_id":match,"timestamp_ms":timestamp_ms(path),"timer_text":text,"seconds_remaining":"" if seconds is None else seconds,"phase":phase_for(seconds,layout,rules),"layout":layout}
            buffer="";position+=1
            if position%10==0:save(rows,args.output)
    finally:save(rows,args.output);cv2.destroyAllWindows()
    print(f"Saved {len(rows)} labels to {args.output}");print(f"Review decisions this run: {len(work)}");return 0


if __name__=="__main__":raise SystemExit(main())
