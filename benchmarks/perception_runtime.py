"""Reproduce the measured M4.5 development probe; see docs/milestone-4.5.md."""
import sys,time,json
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
from scope.mapping import MapConfig
from scope.open_vocab import VerifiedQueryDetector
from scope.query_segmentation import Sam2Segmenter
from scope.common_detector import CommonDetector
from scope.pose_detector import TorchvisionPoseDetector
from scope.perception_pipeline import PerceptionPipeline
from scope.perception_viewer import PerceptionRecorder
from scope.runtime import ModuleConfig,distribution
from scope.semantic_query import SemanticQuery
from scope.interaction import TextCommand
from scope.query_memory import save_query_memory
from scope.backends import hardware_report
out=args.output;out.mkdir(exist_ok=True)
src=TumRgbd(args.tum,180,1,320)
src.selected=src.rgb[355:535];it=iter(src);frames=[];decode=[]
while True:
 start=time.perf_counter()
 try:f=next(it)
 except StopIteration:break
 decode.append((time.perf_counter()-start)*1000);frames.append(f)
query=VerifiedQueryDetector();sam=Sam2Segmenter();common=CommonDetector();pose=TorchvisionPoseDetector()
q=SemanticQuery.from_command(TextCommand('keyboard'))
w=query.query(frames[0],q)
if w.state in ('FOUND','AMBIGUOUS'):sam.segment(frames[0].rgb,w.candidates)
common.detect(frames[0]);pose.detect(frames[0],'front')
cases=[('all',()),('no_rerun',('viewer',)),('no_mapping',('mapping',)),('no_common',('object_detector',)),('no_humans',('human_pose',)),('no_query',('open_vocabulary',)),('core_only',('viewer','mapping','object_detector','human_pose','open_vocabulary','query_memory'))]
report={'hardware':hardware_report(),'decode':distribution(decode),'frames':len(frames),'warm_query':w.summary(),'cases':{},'notes':'180 real frames paced at 30 host Hz; source timestamps retained. Models warmed, shared across sequential cases. Core timings include snapshot/copy/submission. Final file/memory persistence separate. These are scheduling/core responsiveness results, not 30 Hz detector inference.'}
for name,disabled in cases:
 directory=out/name;directory.mkdir(exist_ok=True)
 configs={'human_pose':ModuleConfig(enabled='human_pose' not in disabled,max_hz=3,stale_after_s=2)}
 for module in disabled:configs[module]=ModuleConfig(enabled=False)
 p=PerceptionPipeline(MapConfig.around_frames(frames),lambda:query,lambda:sam,background_detector=common,pose_detector=pose,recorder=PerceptionRecorder(directory),configs=configs)
 p.request('keyboard');ticks=[];output_ages=[];queue=[];start=time.monotonic()
 for i,f in enumerate(frames):
  remaining=start+i/30-time.monotonic()
  if remaining>0:time.sleep(remaining)
  stamp=time.perf_counter();s=p.tick(f,f.timestamp_s);ticks.append((time.perf_counter()-stamp)*1000)
  ages=[t['data_age_s'] for key,t in s['telemetry'].items() if key in ('open_vocabulary','object_detector','pose/front') and t['data_age_s'] is not None]
  output_ages.extend(ages);queue.append(max(t['queue_depth'] for t in s['telemetry'].values()))
 elapsed=time.monotonic()-start
 # Drain recent completed evidence without reading synthetic future frames.
 end=time.monotonic()+1.5
 while time.monotonic()<end:
  p.tick(frames[-1],frames[-1].timestamp_s);time.sleep(.05)
 final=p.snapshot(frames[-1].timestamp_s)
 stamp=time.perf_counter();memory=save_query_memory(p,directory,'tum-public-sitting-static');save_ms=(time.perf_counter()-stamp)*1000
 result={'core_tick':distribution(ticks),'effective_input_hz':len(frames)/elapsed,'wall_s':elapsed,'output_age_ms':distribution(np.array(output_ages)*1000),'max_queue':max(queue),'telemetry':final['telemetry'],'queries':final['queries'],'entities':len(final['entities']),'tracks':len(final['tracks']),'memory':memory,'final_persistence_ms':save_ms,'resources':final['resources']}
 p.close();report['cases'][name]=result
 (out/'performance.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(name,result['core_tick'],'Hz',result['effective_input_hz'],flush=True)
