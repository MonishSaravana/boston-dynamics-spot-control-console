"""Lazy background discovery; independent from foreground text queries."""

import numpy as np

from .detector import _ensure_official_weights
from .objects import ObjectObservation2D
from .query_segmentation import GrabCutSegmenter
from .semantic_query import ObjectCandidate


class CommonDetector:
    name = "torchvision-ssdlite320-mobilenet-v3-coco"
    metadata = {"backend":"torchvision","model":name,"device":"cpu","resolution":320}

    def __init__(self,threshold=.55):
        self.threshold,self.model = threshold,None

    def detect(self,frame):
        if self.model is None:
            import torch
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
            self.torch = torch
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            _ensure_official_weights(torch,weights)
            self.categories = weights.meta["categories"]
            self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to("cpu")
            torch.set_num_threads(min(4,torch.get_num_threads()))
        image = self.torch.from_numpy(frame.rgb.copy()).permute(2,0,1).float()/255.
        with self.torch.inference_mode():found = self.model([image])[0]
        candidates = [ObjectCandidate(tuple(box.numpy().tolist()),float(score),self.categories[int(label)],self.name)
            for box,score,label in zip(found["boxes"],found["scores"],found["labels"])
            if score>=self.threshold and self.categories[int(label)]!="person"]
        GrabCutSegmenter().segment(frame.rgb,candidates)
        return [ObjectObservation2D(f"{frame.frame_id}:common-{i}",frame.frame_id,c.mask,
                {c.detector_match:1.},c.score,self.name) for i,c in enumerate(candidates) if c.mask is not None]
