from __future__ import annotations

import argparse,json
from pathlib import Path
import cv2

try:
    from timer_utils import UNKNOWN,TimerSmoother,format_timer,load_phase_rules,phase_for,predict_timer_image
    from elixir_utils import timestamp_ms
except ModuleNotFoundError:
    from scripts.timer_utils import UNKNOWN,TimerSmoother,format_timer,load_phase_rules,phase_for,predict_timer_image
    from scripts.elixir_utils import timestamp_ms


def parse_args():
    parser=argparse.ArgumentParser(description="Read one timer crop or a chronological directory.");parser.add_argument("input",type=Path);parser.add_argument("--model",type=Path,default=Path("models/timer_recognizer_v1/model.json"));parser.add_argument("--phase-rules",type=Path,default=Path("config/timer_phase_rules.json"));parser.add_argument("--confidence-threshold",type=float);parser.add_argument("--temporal-smoothing",action="store_true");return parser.parse_args()


def main():
    args=parse_args();model=json.loads(args.model.read_text(encoding="utf-8"));rules=load_phase_rules(args.phase_rules);paths=[args.input] if args.input.is_file() else sorted(args.input.glob("*.jpg"),key=timestamp_ms);smoother=TimerSmoother();results=[]
    for index,path in enumerate(paths):
        text,seconds,confidence,layout=predict_timer_image(cv2.imread(str(path)),model,args.confidence_threshold);stamp=timestamp_ms(path) if "_t" in path.name else index*500
        if args.temporal_smoothing:
            seconds=smoother.update(seconds,stamp,layout,confidence);text=UNKNOWN if seconds is None else format_timer(seconds)
        results.append({"image_path":str(path.resolve()),"display_text":text,"seconds_remaining":seconds,"phase":phase_for(seconds,layout,rules),"confidence":confidence})
    print(json.dumps(results,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
