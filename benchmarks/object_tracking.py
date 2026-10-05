"""Reproduce the measured M4.5 development probe; see docs/milestone-4.5.md."""
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
parser=argparse.ArgumentParser(description="Measured local perception benchmark; public or explicitly supplied recordings only")
parser.add_argument("--tum",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise ValueError("Use a new benchmark output path")
args.output.parent.mkdir(parents=True,exist_ok=True)
import numpy as np
from scope.tum import TumRgbd
from scope.open_vocab import VerifiedQueryDetector
from scope.query_segmentation import Sam2Segmenter
from scope.semantic_query import SemanticQuery,box_iou
from scope.interaction import TextCommand
from scope.object_tracking import MaskTracker
from scope.runtime import distribution
src=TumRgbd(args.tum,24,3,320)
frames=list(src);query=SemanticQuery.from_command(TextCommand('keyboard'));model=VerifiedQueryDetector();sam=Sam2Segmenter();truth=[];cost=[]
# Model agreement is a reference, not annotated tracking ground truth.
for f in frames:
 start=time.perf_counter();r=model.query(f,query)
 if r.state in ('FOUND','AMBIGUOUS'):r.timings_ms.update(sam.segment(f.rgb,r.candidates))
 cost.append((time.perf_counter()-start)*1000);truth.append(r);print(f.frame_id,r.state,flush=True)
results={}
for mode,interval in [('detect_once',None),('detect_track_refresh',8)]:
 tracker=MaskTracker();rows=[];ticks=[];detector_calls=0
 for i,(f,r) in enumerate(zip(frames,truth)):
  start=time.perf_counter();tracker.update(f)
  refresh=i==0 or interval is not None and i%interval==0
  if refresh and r.state in ('FOUND','AMBIGUOUS'):
   tracker.initialize(f,query,copy.deepcopy(r.candidates));detector_calls+=1
  flow_cost=(time.perf_counter()-start)*1000;ticks.append(flow_cost)
  active=tracker.active();reference=[c for c in r.candidates if c.mask is not None] if r.state in ('FOUND','AMBIGUOUS') else []
  iou=max((box_iou(t.candidate.box_xyxy,c.box_xyxy) for t in active for c in reference),default=None)
  rows.append({'frame':f.frame_id,'timestamp_s':f.timestamp_s,'tracks':[(t.track_id,str(t.state)) for t in tracker.tracks.values()],'reference_state':str(r.state),'max_box_agreement_iou':iou,'refresh':refresh})
 results[mode]={'tracking_latency':distribution(ticks),'detector_calls':detector_calls,'detector_compute_ms':sum(cost[i] for i in range(len(frames)) if i==0 or interval is not None and i%interval==0),'frames_with_active':sum(any(state=='FOUND' for _,state in row['tracks']) for row in rows),'mean_box_agreement':float(np.mean([r['max_box_agreement_iou'] for r in rows if r['max_box_agreement_iou'] is not None])) if any(r['max_box_agreement_iou'] is not None for r in rows) else None,'rows':rows}
args.output.write_text(json.dumps({'source':'Public TUM fr1 xyz, first 24 usable frames at stride 3','frames':len(frames),'redetection_latency':distribution(cost),'redetection_calls':len(frames),'redetection_compute_ms':sum(cost),'results':results,'reference':[r.summary() for r in truth],'limits':'IoU measures agreement with independently rerun detector, not annotated accuracy. Cold first call retained. Synthetic disappearance/recovery gates tested separately.'},indent=2))
