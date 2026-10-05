"""Discovery-time mask refinement. Bounding boxes never masquerade as masks."""

import time

import numpy as np
from PIL import Image

from .backends import resolve_device, synchronize


class Sam2Segmenter:
    name = "facebook/sam2.1-hiera-tiny"
    revision = "de431c4043854a71d8101e17995dfe596bf101a5"

    def __init__(self, device="auto"):
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError("Install mask support: pip install -e '.[perception]'") from exc
        self.torch, self.device = torch, resolve_device(device)
        self.processor = Sam2Processor.from_pretrained(self.name, revision=self.revision)
        self.model = Sam2Model.from_pretrained(self.name, revision=self.revision,
                                             use_safetensors=True).eval().to(self.device)
        self.metadata = {"backend": "transformers", "model": self.name, "revision": self.revision,
                         "device": self.device, "license": "Apache-2.0"}

    def segment(self, rgb, candidates):
        if not candidates:
            return {"segmentation": 0.}
        start = time.perf_counter()
        inputs = self.processor(images=Image.fromarray(rgb), input_boxes=[[list(c.box_xyxy) for c in candidates]],
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            result = self.model(**inputs, multimask_output=False)
        synchronize(self.device)
        masks = self.processor.post_process_masks(result.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        quality = result.iou_scores.detach().cpu().numpy().reshape(-1)
        for c, mask, q in zip(candidates, masks[:,0].numpy(), quality):
            mask = mask.astype(bool)
            c.mask_quality = float(q)
            c.evidence["segmentation"] = dict(self.metadata)
            if mask.sum() >= 24 and np.isfinite(q) and q >= .5:
                c.mask = mask
            else:
                c.evidence["mask_rejection"] = "Insufficient mask area or predicted mask quality"
        return {"segmentation": (time.perf_counter()-start)*1000}


class GrabCutSegmenter:
    name = "opencv-grabcut"
    metadata = {"backend": "opencv", "model": "GrabCut", "device": "cpu", "license": "Apache-2.0",
                "quality": "Uncalibrated color boundary; not a learned mask"}

    def segment(self, rgb, candidates):
        import cv2
        start = time.perf_counter()
        for c in candidates:
            h, w = rgb.shape[:2]
            x0,y0,x1,y1 = np.rint(c.box_xyxy).astype(int)
            x0,y0,x1,y1 = max(1,x0),max(1,y0),min(w-1,x1),min(h-1,y1)
            if x1-x0 < 4 or y1-y0 < 4:
                c.evidence["mask_rejection"] = "Box too small for GrabCut"
                continue
            labels = np.zeros((h,w),np.uint8)
            cv2.grabCut(rgb, labels, (x0,y0,x1-x0,y1-y0), np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_RECT)
            mask = np.isin(labels,(cv2.GC_FGD,cv2.GC_PR_FGD))
            c.evidence["segmentation"] = dict(self.metadata)
            if mask.sum() >= 24:
                c.mask = mask
            else:
                c.evidence["mask_rejection"] = "No supported color mask"
        return {"segmentation": (time.perf_counter()-start)*1000}
