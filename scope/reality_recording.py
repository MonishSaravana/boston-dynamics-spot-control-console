"""User annotations for public/private RGB recordings, read only after inference."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .perception_sources import image_frame


class RealityRecording:
    def __init__(self,manifest):
        self.path=Path(manifest)
        self.spec=json.loads(self.path.read_text())
        self.frames=self.spec["frames"]
        self.cases=self.spec.get("evaluation_cases",[])
        if not self.cases:
            raise ValueError("Add evaluation_cases annotations to reality.json before benchmarking")
        rooms={}
        for case in self.cases:
            index=case["frame"]
            if not isinstance(index,int) or not 0<=index<len(self.frames):
                raise ValueError("Evaluation frame must be a valid frame index")
            if case["split"] not in ("development","heldout") or not case["room_id"].strip():
                raise ValueError("Each case needs a room_id and development/heldout split")
            if case["room_id"] in rooms and rooms[case["room_id"]]!=case["split"]:
                raise ValueError("A room cannot appear in both development and heldout splits")
            rooms[case["room_id"]]=case["split"]
            if not isinstance(case["present"],bool):
                raise ValueError("Explicit boolean present required; missing annotations are not negatives")
            if case["present"] and not (case.get("boxes_xyxy") or case.get("masks")):
                raise ValueError("Present targets require boxes_xyxy or mask-file annotations")
            if not case["present"] and (case.get("boxes_xyxy") or case.get("masks")):
                raise ValueError("Absent target cannot have target annotations")
        pairs=[(c["frame"],c["query"]) for c in self.cases]
        if len(set(pairs))!=len(pairs):raise ValueError("Duplicate frame/query annotation")

    def frame(self,index):
        row=self.frames[index]
        relative=lambda value:self.path.parent/value if value else None
        return image_frame(relative(row["image"]),relative(row.get("depth")),relative(row.get("calibration")))

    def truth(self,index,phrase,shape):
        case=next(c for c in self.cases if c["frame"]==index and c["query"]==phrase)
        masks=[np.asarray(Image.open(self.path.parent/name).convert("L"))>0 for name in case.get("masks",[])]
        if any(m.shape!=shape for m in masks):raise ValueError("Annotation mask shape differs from image")
        boxes=list(case.get("boxes_xyxy",[]))
        if not boxes:
            for m in masks:
                yy,xx=np.nonzero(m)
                if len(xx):boxes.append((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
        h,w=shape
        for box in boxes:
            if len(box)!=4 or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
                raise ValueError("Annotation box outside image or not x0,y0,x1,y1")
        return masks,boxes

    def close(self):pass
