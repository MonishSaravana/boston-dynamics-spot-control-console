"""Room-disjoint real RGB-D evaluation; labels never enter inference."""

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .backends import hardware_report
from .interaction import TextCommand
from .nyu_source import NyuDataset
from .objects import ObjectObservation2D,project_object
from .runtime import distribution
from .semantic_query import ObjectCandidate,QueryResult,SemanticQuery,box_iou,decide


VOCABULARY = {
    "sofa":["sofa"],"couch":["sofa"],"chair":["chair"],"backpack":["backpack"],
    "bag":["bag","backpack"],"water bottle":["bottle"],"monitor":["monitor"],"screen":["monitor"],
    "notebook":["notebook"],"keyboard":["keyboard"],"trash can":["garbage bin"],"garbage bin":["garbage bin"],
    "pillow":["pillow"],"lamp":["lamp"],"bookshelf":["bookshelf"],"cabinet":["cabinet"],
    "whiteboard":["whiteboard"],"power strip":["power strip"],"whiteboard eraser":["whiteboard eraser"],
    "fire extinguisher":["fire extinguisher"],"door":["door"],"charging brick":["charger"],
    "red toolbox":["toolbox"],"flarblenox":[],
}
NON_COCO = {"pillow","lamp","bookshelf","cabinet","whiteboard","power strip","whiteboard eraser",
            "fire extinguisher","door","charging brick","notebook"}


def prepare_spec(dataset,output):
    source=NyuDataset(dataset)
    dev={"office_0003","living_room_0000"}
    rooms=[]
    for scene in sorted(dev):
        index=source.scenes.index(scene)
        rooms.append({"index":index,"scene":scene,"split":"development","type":source.types[index]})
    for kind in ("living_room","classroom","computer_lab","bedroom","kitchen","student_lounge"):
        index=next(i for i,t in enumerate(source.types) if t==kind and source.scenes[i] not in dev)
        rooms.append({"index":index,"scene":source.scenes[index],"split":"heldout","type":kind})
    spec={"format_version":1,"dataset":"official NYU Depth V2 labeled mat",
          "scene_selection":"First labeled frame of each specified type excluding development rooms; selected before model comparison",
          "rooms":rooms,"queries":VOCABULARY,"non_coco":sorted(NON_COCO),
          "metrics":"box IoU >= 0.5 against any annotated instance; mask IoU; raw aligned depth projection. Absent label is annotation-level absence, not exhaustive proof beyond the image.",
          "model_spec":{"grounding-dino":{"threshold":.4,"size":512},"owlv2":{"threshold":.4,"size":512}},
          "limitations":"Rooms held out from SCOPE development; unknown overlap with upstream pretraining. Water bottle is evaluated as bottle; contents/brand not verified. Red toolbox negatives require no annotated toolbox."}
    output=Path(output)
    if output.exists():raise ValueError("Benchmark spec already exists; preserve frozen split")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(spec,indent=2)+"\n")
    source.close()
    return spec


def ground_truth(source,index,labels):
    image_labels,instances=source.labels(index)
    masks=[]
    for label in labels:
        if label not in source.names:continue
        selected=image_labels==source.names.index(label)+1
        for identity in np.unique(instances[selected]):
            mask=selected&(instances==identity)
            if mask.sum()>=12:masks.append(mask)
    return masks


def score_case(frame,result,masks,boxes=None):
    gt_boxes=list(boxes or [])
    for mask in masks:
        yy,xx=np.nonzero(mask)
        if len(xx):gt_boxes.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
    supported=result.state in ("FOUND","AMBIGUOUS")
    best_box=max((box_iou(c.box_xyxy,g) for c in result.candidates for g in gt_boxes),default=0.)
    mask_iou=max((float((c.mask&m).sum()/max(1,(c.mask|m).sum())) for c in result.candidates if c.mask is not None for m in masks),default=None)
    projected=0
    for i,c in enumerate(result.candidates if supported else []):
        if c.mask is None:continue
        detection=ObjectObservation2D(f'{frame.frame_id}:benchmark-{i}',frame.frame_id,c.mask,
            {result.query.normalized:1.},c.score,c.model)
        if project_object(frame,detection) is not None:projected+=1
    return {"present":bool(gt_boxes),"gt_instances":len(masks) if masks else len(gt_boxes),"state":str(result.state),
        "success":bool(gt_boxes and supported and best_box>=.5),"false_positive":bool(not gt_boxes and supported),
        "best_box_iou":best_box,"best_mask_iou":mask_iou,"projected_candidates":projected,
        "candidate_false_positives":sum(max((box_iou(c.box_xyxy,g) for g in gt_boxes),default=0.)<.5 for c in result.candidates) if supported else 0}


def aggregate(rows):
    present=[r for r in rows if r["metrics"]["present"]]
    absent=[r for r in rows if not r["metrics"]["present"]]
    return {"cases":len(rows),"present_cases":len(present),"absent_cases":len(absent),
        "query_success":sum(r["metrics"]["success"] for r in present)/len(present) if present else None,
        "false_positive_rate_absent":sum(r["metrics"]["false_positive"] for r in absent)/len(absent) if absent else None,
        "states":{state:sum(r["metrics"]["state"]==state for r in rows) for state in sorted({r["metrics"]["state"] for r in rows})},
        "mean_matched_mask_iou":float(np.mean([r["metrics"]["best_mask_iou"] for r in present if r["metrics"]["success"] and r["metrics"]["best_mask_iou"] is not None])) if any(r["metrics"]["success"] and r["metrics"]["best_mask_iou"] is not None for r in present) else None,
        "depth_projection_success_cases":sum(r["metrics"]["success"] and r["metrics"]["projected_candidates"]>0 for r in present),
        "latency_ms":distribution([r["total_ms"] for r in rows])}


def benchmark(dataset,spec_path,output,split="heldout",backends=("closed","grounding-dino","owlv2"),device="auto",manifest=None):
    from .detector import TorchvisionMaskDetector
    from .open_vocab import OpenVocabularyDetector,corroborate
    from .query_segmentation import Sam2Segmenter,GrabCutSegmenter
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("Use an empty benchmark output")
    output.mkdir(parents=True,exist_ok=True)
    if manifest:
        from .reality_recording import RealityRecording
        source=RealityRecording(manifest)
        spec={"rooms":[],"queries":{},"non_coco":sorted(NON_COCO),
              "model_spec":{"grounding-dino":{"threshold":.4,"size":512},"owlv2":{"threshold":.4,"size":512}}}
        for case in source.cases:
            room=next((r for r in spec["rooms"] if r["index"]==case["frame"]),None)
            if room is None:
                room={"index":case["frame"],"scene":case["room_id"],"split":case["split"],"type":"user recording","queries":[]}
                spec["rooms"].append(room)
            if room["scene"]!=case["room_id"] or room["split"]!=case["split"]:
                raise ValueError("All annotations for one frame need the same room and split")
            room["queries"].append(case["query"])
        spec_path=manifest
    else:
        if not spec_path:raise ValueError("NYU evaluation requires --spec with a frozen split")
        spec=json.loads(Path(spec_path).read_text())
        source=NyuDataset(dataset)
    rooms=[r for r in spec["rooms"] if r["split"]==split]
    if not rooms:raise ValueError("No rooms in this split")
    rows=[];models={}
    for backend in backends:
        if backend=="closed":
            model=TorchvisionMaskDetector()
            models[backend]={"backend":"torchvision","model":model.name,"device":"cpu","resolution":480,"threshold":.55}
        else:
            cfg=spec["model_spec"][backend]
            model=OpenVocabularyDetector(backend,device,cfg["size"],cfg["threshold"])
            models[backend]=model.metadata
        segmenter=Sam2Segmenter(device) if backend=="grounding-dino" else None
        verifier=OpenVocabularyDetector("owlv2",device,512,.15,.075) if backend=="grounding-dino" else None
        if verifier:
            models[backend+"+verified+sam2"]={**model.metadata,"acceptance_threshold":.45,
                "verifier":verifier.metadata,"minimum_box_iou":.3,"segmenter":segmenter.metadata}
        for room in rooms:
            frame=source.frame(room["index"])
            closed=None
            if backend=="closed":
                start=time.perf_counter();closed=model.detect(frame);closed_ms=(time.perf_counter()-start)*1000
            phrases=((q,None) for q in room["queries"]) if manifest else spec["queries"].items()
            for phrase,gt_labels in phrases:
                query=SemanticQuery.from_command(TextCommand(phrase))
                start=time.perf_counter()
                if backend=="closed":
                    candidates=[ObjectCandidate(tuple(o.box_xyxy),o.confidence,o.label,model.name,o.mask)
                        for o in closed if any(o.label==a or a.endswith(" "+o.label) for a in query.alternatives)]
                    state,reason,candidates=decide(candidates,.55)
                    result=QueryResult(query,state,reason,candidates,frame.timestamp_s,frame.frame_id,models[backend],{"closed_detection":closed_ms})
                    total=closed_ms
                else:
                    result=model.query(frame,query);total=(time.perf_counter()-start)*1000
                # Ground truth is read after detector output and never passed to a model.
                masks,boxes=source.truth(room["index"],phrase,frame.rgb.shape[:2]) if manifest else (ground_truth(source,room["index"],gt_labels),None)
                variants=[(backend,result,total)]
                if segmenter and result.state in ("FOUND","AMBIGUOUS"):
                    checked=copy.deepcopy(result)
                    stamp=time.perf_counter()
                    if any(c.score>=.45 for c in checked.candidates):
                        v=verifier.query(frame,query);checked=corroborate(checked,v)
                    else:
                        from .semantic_query import QueryState
                        checked.state,checked.reason=QueryState.LOW_CONFIDENCE,"Primary proposal below verified acceptance threshold"
                    verify_ms=(time.perf_counter()-stamp)*1000
                    checked.timings_ms["verification"]=verify_ms
                    checked_cost=total+verify_ms
                    if checked.state in ("FOUND","AMBIGUOUS"):
                        extra=segmenter.segment(frame.rgb,checked.candidates)
                        checked.timings_ms.update(extra);checked_cost+=extra["segmentation"]
                    variants.append((backend+"+verified+sam2",checked,checked_cost))
                    refined=copy.deepcopy(result)
                    durations=segmenter.segment(frame.rgb,refined.candidates)
                    refined.timings_ms.update(durations)
                    variants.append((backend+"+sam2",refined,total+durations["segmentation"]))
                    color=copy.deepcopy(result);cost=GrabCutSegmenter().segment(frame.rgb,color.candidates)
                    color.timings_ms.update(cost)
                    variants.append((backend+"+grabcut",color,total+cost["segmentation"]))
                elif backend=="grounding-dino":
                    variants.extend([(backend+"+sam2",result,total),(backend+"+grabcut",result,total),
                                     (backend+"+verified+sam2",result,total)])
                for variant,value,cost in variants:
                    rows.append({"backend":variant,"room":room,"query":phrase,"non_coco":phrase in spec["non_coco"],
                        "metrics":score_case(frame,value,masks,boxes),"result":value.summary(),"total_ms":cost})
            print(backend,room["scene"],"complete",flush=True)
            (output/"cases.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
        del model,segmenter,verifier
        try:
            import torch
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        except ImportError:pass
    names=sorted({r["backend"] for r in rows})
    report={"split":split,"frozen_spec_sha256":hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        "hardware":hardware_report(),"models":models,"rooms":rooms,
        "summary":{name:aggregate([r for r in rows if r["backend"]==name]) for name in names},
        "non_coco":{name:aggregate([r for r in rows if r["backend"]==name and r["non_coco"]]) for name in names},
        "per_room":{name:{room["scene"]:aggregate([r for r in rows if r["backend"]==name and r["room"]["scene"]==room["scene"]]) for room in rooms} for name in names},
        "notes":"Cold first calls included. Closed detection runs once per image and cost is repeated per query, not summed as independent inference. SAM2/GrabCut share the same query boxes; supplied labels only score outputs. Raw scores and negatives retained."}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    source.close()
    return report
