"""Optional, local text-conditioned detectors. No hosted inference or LLM."""

import time

import numpy as np
from PIL import Image

from .backends import resolve_device, synchronize
from .semantic_query import ObjectCandidate, QueryResult, decide


MODELS = {
    "grounding-dino": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "owlv2": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
}


class OpenVocabularyDetector:
    def __init__(self, backend="grounding-dino", device="auto", size=512,
                 threshold=.4, low_threshold=.15):
        if backend not in MODELS or size < 224 or not 0 < low_threshold <= threshold < 1:
            raise ValueError("Invalid detector, resolution, or thresholds")
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise RuntimeError("Install local query support: pip install -e '.[perception]'") from exc
        self.torch, self.backend = torch, backend
        self.device = resolve_device(device)
        self.size, self.threshold, self.low_threshold = size, threshold, low_threshold
        torch.set_num_threads(min(4, torch.get_num_threads()))
        repo, revision = MODELS[backend]
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision, use_fast=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            repo, revision=revision, use_safetensors=True).eval().to(self.device)
        self.metadata = {"backend": "transformers", "model": repo, "revision": revision,
                         "device": self.device, "dtype": "float32", "resolution": size,
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "weight_bytes": sum(p.numel()*p.element_size() for p in self.model.parameters()),
                         "license": "Apache-2.0", "transformers": __import__("transformers").__version__}
        self.metadata.update(acceptance_threshold=threshold,weak_threshold=low_threshold,text_threshold=.25)
        self.name = repo

    def query(self, frame, query):
        start = time.perf_counter()
        image = Image.fromarray(frame.rgb)
        if self.backend == "grounding-dino":
            caption = ". ".join(query.alternatives)+"."
            inputs = self.processor(images=image, text=caption, return_tensors="pt",
                                    size={"shortest_edge": self.size, "longest_edge": self.size*2}).to(self.device)
        else:
            inputs = self.processor(images=image, text=[list(query.alternatives)], return_tensors="pt",
                                    size={"height": self.size, "width": self.size}).to(self.device)
        synchronize(self.device)
        pre_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, **({"interpolate_pos_encoding": True} if self.backend == "owlv2" else {}))
        synchronize(self.device)
        infer_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        if self.backend == "grounding-dino":
            found = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
                threshold=self.low_threshold, text_threshold=.25,
                target_sizes=[frame.rgb.shape[:2]])[0]
            labels = found.get("text_labels", found.get("labels"))
        else:
            found = self.processor.post_process_object_detection(outputs, threshold=self.low_threshold,
                target_sizes=self.torch.tensor([frame.rgb.shape[:2]], device=self.device))[0]
            labels = [query.alternatives[int(i)] for i in found["labels"].cpu()]
        candidates = []
        h, w = frame.rgb.shape[:2]
        for box, score, label in zip(found["boxes"].detach().cpu().numpy(), found["scores"].detach().cpu().numpy(), labels):
            box = np.clip(box, [0,0,0,0], [w,h,w,h])
            if np.isfinite(box).all() and box[2] > box[0]+2 and box[3] > box[1]+2:
                candidates.append(ObjectCandidate(tuple(float(x) for x in box), float(score), str(label), self.name))
        state, reason, candidates = decide(candidates, self.threshold, self.low_threshold)
        return QueryResult(query, state, reason, candidates, frame.timestamp_s, frame.frame_id,
                           dict(self.metadata), {"preprocess": pre_ms, "open_vocab": infer_ms,
                           "postprocess": (time.perf_counter()-start)*1000})


def corroborate(result,verification,min_iou=.3,min_primary_score=.45):
    """Detector agreement is extra evidence, never proof of correctness."""
    from .semantic_query import QueryState,box_iou
    accepted,rejected=[],[]
    for candidate in result.candidates:
        matches=[(box_iou(candidate.box_xyxy,c.box_xyxy),c) for c in verification.candidates]
        best=max(matches,key=lambda x:x[0]) if matches else None
        agreed=bool(candidate.score>=min_primary_score and best and best[0]>=min_iou and best[1].score>=.15)
        candidate.evidence["verification"]={"model":verification.metadata.get("model"),
            "agreed":agreed,"box_iou":best[0] if best else None,
            "score":best[1].score if best else None,"minimum_box_iou":min_iou,
            "minimum_primary_score":min_primary_score}
        (accepted if agreed else rejected).append(candidate)
    result.metadata["rejected_proposals"]=[c.summary() for c in rejected]
    if not accepted:
        result.state,result.reason=QueryState.LOW_CONFIDENCE,"Independent detector did not corroborate the proposed region"
        # Retain rejected regions for debugging; they never enter 3D/tracking.
    else:
        result.candidates=accepted
        result.state=QueryState.AMBIGUOUS if len(accepted)>1 or result.state==QueryState.AMBIGUOUS else QueryState.FOUND
        result.reason="Multiple regions or unresolved alternatives; no automatic target choice" if result.state==QueryState.AMBIGUOUS else "Two text-conditioned detectors support one region; evidence remains uncalibrated"
    return result


class VerifiedQueryDetector:
    def __init__(self,device="auto",size=512,threshold=.45):
        self.primary=OpenVocabularyDetector("grounding-dino",device,size,threshold)
        self.verifier=OpenVocabularyDetector("owlv2",device,512,.15,.075)
        self.name=self.primary.name+"+owlv2-verification"
        self.metadata={**self.primary.metadata,"verification":{**self.verifier.metadata,"minimum_box_iou":.3}}

    def query(self,frame,query):
        from .semantic_query import QueryState
        result=self.primary.query(frame,query)
        result.metadata=dict(self.metadata)
        if result.state in (QueryState.FOUND,QueryState.AMBIGUOUS):
            start=time.perf_counter()
            verification=self.verifier.query(frame,query)
            result=corroborate(result,verification,min_primary_score=self.primary.threshold)
            result.timings_ms["verification"]=(time.perf_counter()-start)*1000
        return result
