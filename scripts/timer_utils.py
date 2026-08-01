from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


TIMER_RE=re.compile(r"^(\d):([0-5]\d)$")
UNKNOWN="unknown"
CHAR_BOXES=((19,29,47,78),(58,29,84,78),(82,29,110,78))
COLON_BOX=(46,34,62,74)


def parse_timer_text(text:str)->int:
    match=TIMER_RE.fullmatch(text.strip())
    if not match:raise ValueError(f"Timer must use M:SS, got {text!r}")
    return int(match.group(1))*60+int(match.group(2))


def format_timer(seconds:int)->str:
    if not 0<=seconds<=599:raise ValueError("Timer seconds must be from 0 to 599")
    return f"{seconds//60}:{seconds%60:02d}"


def timer_layout(image:np.ndarray)->str:
    resized=cv2.resize(image,(125,83),interpolation=cv2.INTER_AREA);hsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)
    header=hsv[:31]
    red=(((header[...,0]<=12)|(header[...,0]>=170))&(header[...,1]>80)&(header[...,2]>80)).mean()
    return "overtime" if red>.12 else "regulation"


def _centered_glyph(mask:np.ndarray)->np.ndarray:
    points=cv2.findNonZero(mask)
    canvas=np.zeros((30,18),np.uint8)
    if points is None:return canvas.astype(np.float32).ravel()
    x,y,w,h=cv2.boundingRect(points);glyph=mask[y:y+h,x:x+w]
    scale=min(14/max(w,1),26/max(h,1));glyph=cv2.resize(glyph,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    top=(30-glyph.shape[0])//2;left=(18-glyph.shape[1])//2;canvas[top:top+glyph.shape[0],left:left+glyph.shape[1]]=glyph
    return canvas.astype(np.float32).ravel()/255


def character_features(image:np.ndarray)->list[np.ndarray]:
    resized=cv2.resize(image,(125,83),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    features=[]
    for x1,y1,x2,y2 in CHAR_BOXES:
        crop=cv2.resize(gray[y1:y2,x1:x2],(18,30),interpolation=cv2.INTER_AREA).astype(np.float32)
        crop=(crop-crop.mean())/max(float(crop.std()),1.0);features.append((crop.ravel()/max(float(np.linalg.norm(crop)),1e-6)).astype(np.float32))
    return features


def colon_score(image:np.ndarray)->float:
    hsv=cv2.cvtColor(cv2.resize(image,(125,83)),cv2.COLOR_BGR2HSV);x1,y1,x2,y2=COLON_BOX;crop=hsv[y1:y2,x1:x2]
    white=((crop[...,2]>155)&(crop[...,1]<105)).astype(np.uint8)
    components=cv2.connectedComponentsWithStats(white,8)[2]
    dots=sum(2<=area<=45 for area in components[1:,cv2.CC_STAT_AREA])
    return 1.0 if dots>=2 else 0.0


def train_template_model(samples:list[tuple[np.ndarray,str]])->dict[str,object]:
    grouped=[{} for _ in range(3)]
    for image,text in samples:
        digits=text.replace(":","");features=character_features(image)
        for position,(digit,feature) in enumerate(zip(digits,features)):grouped[position].setdefault(digit,[]).append(feature)
    required=[set("012"),set("012345"),set("0123456789")]
    missing={str(position):sorted(required[position]-set(grouped[position])) for position in range(3) if required[position]-set(grouped[position])}
    if missing:raise ValueError(f"Training labels lack position-specific digits: {missing}")
    exemplars=[]
    for position in range(3):
        exemplars.append({digit:[row.tolist() for row in rows] for digit,rows in grouped[position].items()})
    return {"method":"fixed_position_timer_exemplars_v3","position_exemplars":exemplars,"unknown_threshold":.5}


def predict_timer_image(image:np.ndarray,model:dict[str,object],threshold:float|None=None):
    digits=[];scores=[]
    for position,feature in enumerate(character_features(image)):
        ranked=[]
        for digit,rows in model["position_exemplars"][position].items():
            similarity=max(float(feature@np.asarray(row,np.float32)) for row in rows);ranked.append((similarity,digit))
        ranked.sort(reverse=True);best,digit=ranked[0];second=ranked[1][0];quality=float(np.clip((best-.35)/.65,0,1));margin=float(np.clip((best-second)/.18,0,1))
        digits.append(digit);scores.append(float(np.sqrt(quality*margin)))
    confidence=float(min(scores)*(.9+.1*colon_score(image)));text=f"{digits[0]}:{digits[1]}{digits[2]}"
    valid=int(digits[1])<=5;cutoff=float(model.get("unknown_threshold",.5) if threshold is None else threshold)
    if confidence<cutoff or not valid:return UNKNOWN,None,confidence,timer_layout(image)
    return text,parse_timer_text(text),confidence,timer_layout(image)


def load_phase_rules(path:Path)->dict: return json.loads(path.read_text(encoding="utf-8"))


def phase_for(seconds:int|None,layout:str,rules:dict)->str:
    if seconds is None or layout not in rules:return UNKNOWN
    for rule in sorted(rules[layout],key=lambda row:int(row["minimum_seconds"]),reverse=True):
        if seconds>=int(rule["minimum_seconds"]):return str(rule["phase"])
    return UNKNOWN


class TimerSmoother:
    def __init__(self):self.seconds=None;self.timestamp=None;self.layout=None;self.clock_offset=None
    def update(self,seconds:int|None,timestamp_ms:int,layout:str,confidence:float=.5)->int|None:
        if seconds is None:return self.seconds
        if self.seconds is None:
            self.seconds,self.timestamp,self.layout=seconds,timestamp_ms,layout;self.clock_offset=seconds+timestamp_ms/1000;return seconds
        expected=max(0,round(float(self.clock_offset)-timestamp_ms/1000))
        phase_transition=self.layout=="regulation" and layout=="overtime" and expected<=1
        if phase_transition:self.clock_offset=seconds+timestamp_ms/1000
        elif (seconds>self.seconds or abs(seconds-expected)>2) and confidence<.8:seconds=expected
        elif confidence>=.8:self.clock_offset=seconds+timestamp_ms/1000
        self.seconds,self.timestamp,self.layout=seconds,timestamp_ms,layout;return seconds
