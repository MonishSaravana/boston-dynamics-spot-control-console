"""Reproduce the measured M4.5 development probe; see docs/milestone-4.5.md."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
parser=argparse.ArgumentParser(description="Measured local perception benchmark; public or explicitly supplied recordings only")
parser.add_argument("--nyu-mat",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise ValueError("Use a new benchmark output path")
args.output.parent.mkdir(parents=True,exist_ok=True)
import numpy as np
from scope.nyu_source import NyuDataset
from scope.open_vocab import OpenVocabularyDetector
from scope.interaction import TextCommand
from scope.semantic_query import SemanticQuery
from scope.reality_benchmark import ground_truth,score_case
from scope.common_detector import CommonDetector
from scope.detector import TorchvisionMaskDetector
from scope.runtime import distribution
from scope.backends import hardware_report,synchronize
source=NyuDataset(args.nyu_mat)
queries=[(49,'sofa',['sofa']),(49,'pillow',['pillow']),(49,'lamp',['lamp']),(2,'chair',['chair']),(2,'door',['door']),(2,'red toolbox',['toolbox'])]
rows=[]
for backend,device,size in [('grounding-dino','cpu',512),('grounding-dino','mps',512),('grounding-dino','mps',800),('owlv2','cpu',512),('owlv2','mps',512),('owlv2','mps',768)]:
 model=OpenVocabularyDetector(backend,device,size,.45 if backend=='grounding-dino' else .4)
 warm=model.query(source.frame(49),SemanticQuery.from_command(TextCommand('sofa')))
 times=[];results=[]
 for index,phrase,labels in queries:
  frame=source.frame(index);start=time.perf_counter();r=model.query(frame,SemanticQuery.from_command(TextCommand(phrase)));elapsed=(time.perf_counter()-start)*1000;times.append(elapsed)
  results.append({'index':index,'phrase':phrase,'metrics':score_case(frame,r,ground_truth(source,index,labels)),'result':r.summary(),'wall_ms':elapsed})
 row={'backend':backend,'device':device,'size':size,'latency':distribution(times),'quality':results,'metadata':model.metadata};rows.append(row);print(backend,device,size,row['latency'],flush=True)
 del model
 import torch
 torch.mps.empty_cache()
common=[]
for factory in [CommonDetector,TorchvisionMaskDetector]:
 model=factory();model.detect(source.frame(49));times=[];outputs=[]
 for index in [49,2,49,2,49,2]:
  frame=source.frame(index);start=time.perf_counter();found=model.detect(frame);elapsed=(time.perf_counter()-start)*1000;times.append(elapsed)
  outputs.append({'index':index,'objects':[{'label':o.label,'score':o.confidence,'box':o.box_xyxy,'mask_pixels':int(o.mask.sum())} for o in found]})
 common.append({'model':model.name,'latency':distribution(times),'outputs':outputs});print(model.name,common[-1]['latency'],flush=True)
 del model
args.output.write_text(json.dumps({'hardware':hardware_report(),'resolution':rows,'common':common},indent=2))
