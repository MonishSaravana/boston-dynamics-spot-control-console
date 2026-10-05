"""Ordinary object phrases share one local, modular perception path."""

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw

from .backends import hardware_report
from .mapping import MapConfig
from .perception_pipeline import PerceptionPipeline
from .runtime import LatestWorker,ModuleConfig,distribution


def register_perception_commands(commands):
    commands.add_parser("hardware",help="Available inference devices and measured host resources")
    p=commands.add_parser("prepare-reality",help="Freeze room-disjoint NYU scene IDs and query vocabulary")
    p.add_argument("--nyu-mat",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p=commands.add_parser("benchmark-reality",help="Real held-out room detector/mask/depth comparison")
    benchmark_source=p.add_mutually_exclusive_group(required=True)
    benchmark_source.add_argument("--nyu-mat",type=Path);benchmark_source.add_argument("--manifest",type=Path)
    p.add_argument("--spec",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--split",choices=("development","heldout"),default="heldout")
    p.add_argument("--backend",action="append",choices=("closed","grounding-dino","owlv2"))
    p.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    for name in ("query-object","perceive"):
        p = commands.add_parser(name,help="Query ordinary objects in an image, recording, or camera")
        if name=="query-object":p.add_argument("phrase")
        else:p.add_argument("--query",action="append",default=[])
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--image",type=Path)
        source.add_argument("--nyu-mat",type=Path)
        source.add_argument("--episode",type=Path)
        source.add_argument("--tum",type=Path)
        source.add_argument("--video",type=Path)
        source.add_argument("--webcam",type=int)
        p.add_argument("--nyu-index",type=int,default=49)
        p.add_argument("--depth",type=Path)
        p.add_argument("--calibration",type=Path)
        p.add_argument("--world-frame")
        p.add_argument("--start-frame",type=int,default=0)
        p.add_argument("--frames",type=int,default=100)
        p.add_argument("--width",type=int,default=640)
        p.add_argument("--duration",type=float,default=30)
        p.add_argument("--timeout",type=float,default=120)
        p.add_argument("--backend",choices=("grounding-dino","owlv2"),default="grounding-dino")
        p.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
        p.add_argument("--model-size",type=int,default=512)
        p.add_argument("--threshold",type=float,default=.45)
        p.add_argument("--unverified",action="store_true",help="Keep raw single-model proposals for comparison; default DINO queries require OWLv2 agreement")
        p.add_argument("--segmentation",choices=("sam2","grabcut","none"),default="sam2")
        p.add_argument("--background",action="store_true")
        p.add_argument("--humans",action="store_true")
        p.add_argument("--interactive",action="store_true")
        p.add_argument("--disable",action="append",default=[])
        p.add_argument("--rate",action="append",default=[])
        p.add_argument("--no-rerun",action="store_true")
        p.add_argument("--output",type=Path)
    p = commands.add_parser("capture-reality",help="Explicit RGB webcam capture or phone/video import")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--webcam",type=int);source.add_argument("--video",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=100);p.add_argument("--hz",type=float,default=10)
    p.add_argument("--width",type=int,default=640)


class InputDriver:
    def __init__(self,frame,reader=None,close=None,initial_decode_ms=None):
        self.current,self.reader,self.close_source = frame,reader,close
        self.initial_decode_ms=initial_decode_ms
        self.last_receive = time.monotonic()
        self.worker = LatestWorker(lambda _:reader()) if reader else None
        self.pending,self.finished = False,False
        self.decode_ms = []

    def poll(self,runtime=None):
        if self.worker:
            result = self.worker.poll()
            if result:
                _,value,error,duration = result
                self.pending = False;self.decode_ms.append(duration)
                if error:raise RuntimeError(error)
                if value is None:self.finished = True
                else:
                    self.current,self.last_receive = value,time.monotonic()
                    if runtime:
                        runtime.observe("decode",value.timestamp_s,duration,metadata={"backend":"recording/camera reader","device":"cpu"})
            if not self.pending and not self.finished:
                self.worker.submit(True);self.pending = True
        return self.current

    def now(self):
        return self.current.timestamp_s+time.monotonic()-self.last_receive

    def close(self):
        done = self.worker.close(timeout_s=5.) if self.worker else True
        if done and self.close_source:self.close_source()


def load_source(args):
    from .perception_sources import image_frame,VideoSource
    close,reader = None,None
    start = time.perf_counter()
    if args.image:
        frame = image_frame(args.image,args.depth,args.calibration)
        meta = {"source":str(args.image),"acquisition_timestamp_s":None,"world_frame":args.world_frame or "file-camera:"+str(args.image.resolve()),
                "clock":"host file load; original capture time unavailable","intrinsics":"supplied" if args.depth else None}
    elif args.nyu_mat:
        from .nyu_source import NyuDataset
        source = NyuDataset(args.nyu_mat)
        frame = source.frame(args.nyu_index,time.monotonic());meta = source.metadata(args.nyu_index)
        close = source.close
    elif args.tum or args.episode:
        if args.tum:
            from .tum import TumRgbd
            source = TumRgbd(args.tum,args.frames,1,args.width)
            source.selected = source.rgb[args.start_frame:args.start_frame+args.frames]
            iterator = iter(source)
            path = args.tum
        else:
            from .storage import load_episode
            iterator = iter(load_episode(args.episode)[args.start_frame:args.start_frame+args.frames])
            path = args.episode
        frame = next(iterator,None)
        if frame is None:raise ValueError("No usable frames in this recording")
        reader = lambda:next(iterator,None)
        meta = {"source":str(path),"world_frame":args.world_frame or "recording:"+str(path.resolve()),
                "poses":"supplied recording poses","clock":"recorded source seconds"}
    else:
        source = VideoSource(args.webcam if args.webcam is not None else args.video,args.width)
        frame = source.read();reader,close = source.read,source.close
        if frame is None:raise ValueError("Selected input returned no image")
        meta = {"source":"webcam" if args.webcam is not None else str(args.video),"depth":None,"intrinsics":None,
                "acquisition_timestamp_s":None,"world_frame":"RGB-only","clock":"host receive" if args.webcam is not None else "video presentation seconds"}
    meta["initial_decode_ms"] = (time.perf_counter()-start)*1000
    return InputDriver(frame,reader,close,meta["initial_decode_ms"]),meta


def make_pipeline(args,driver,output):
    from .common_detector import CommonDetector
    from .open_vocab import OpenVocabularyDetector,VerifiedQueryDetector
    from .query_segmentation import Sam2Segmenter,GrabCutSegmenter
    from .perception_viewer import PerceptionRecorder
    if args.frames<1 or args.duration<=0 or args.timeout<=0 or args.width<64:
        raise ValueError("Positive frames, duration, timeout and width at least 64 required")
    frame = driver.current
    map_config = MapConfig.around_frames([frame]) if np.isfinite(frame.depth_m).any() else MapConfig.synthetic()
    configs = {"mapping":ModuleConfig(enabled=bool(np.isfinite(frame.depth_m).any()),max_hz=5),
               "human_pose":ModuleConfig(enabled=args.humans,max_hz=3,stale_after_s=2),
               "object_detector":ModuleConfig(enabled=args.background,max_hz=.5,stale_after_s=3),
               "viewer":ModuleConfig(enabled=not args.no_rerun,max_hz=1 if args.interactive else 5,stale_after_s=3),
               "segmentation":ModuleConfig(enabled=args.segmentation!="none",stale_after_s=3)}
    class LazyPose:
        name="torchvision-keypointrcnn-resnet50-fpn-coco-v1"
        metadata={"backend":"torchvision","model":name,"device":"cpu","resolution":640}
        model=None
        def detect(self,f,c):
            if self.model is None:
                from .pose_detector import TorchvisionPoseDetector
                self.model=TorchvisionPoseDetector()
            return self.model.detect(f,c)
    pose=LazyPose()
    segmenter = (lambda:Sam2Segmenter(args.device)) if args.segmentation=="sam2" else GrabCutSegmenter if args.segmentation=="grabcut" else None
    query_factory=(lambda:VerifiedQueryDetector(args.device,args.model_size,args.threshold)) if args.backend=="grounding-dino" and not args.unverified else (
        lambda:OpenVocabularyDetector(args.backend,args.device,args.model_size,args.threshold))
    p = PerceptionPipeline(map_config,query_factory,segmenter,
        background_detector=CommonDetector(),pose_detector=pose,
        recorder=None if args.no_rerun else PerceptionRecorder(output),configs=configs)
    p.runtime.state("pose/front").metadata=dict(pose.metadata)
    p.runtime.state("human_pose").metadata=dict(pose.metadata)
    p.runtime.configure("decode",ModuleConfig(stale_after_s=2))
    if driver.initial_decode_ms is not None:
        p.runtime.observe("decode",frame.timestamp_s,driver.initial_decode_ms,
            metadata={"backend":"initial source open/decode","device":"cpu"})
    aliases={"memory":"query_memory","rerun":"viewer","human_pose":"human_pose","semantics":"object_detector"}
    for name in args.disable:
        name=aliases.get(name,name)
        if name not in p.runtime.modules:raise ValueError(f"Unknown module {name}")
        from dataclasses import replace
        p.configure(name,replace(p.runtime.state(name).config,enabled=False))
    for setting in args.rate:
        name,hz=setting.split("=",1);name=aliases.get(name,name)
        if name not in p.runtime.modules:raise ValueError(f"Unknown module {name}")
        from dataclasses import replace
        p.configure(name,replace(p.runtime.state(name).config,max_hz=float(hz)))
    return p


def overlay(frame,snapshot):
    rgba = Image.fromarray(frame.rgb).convert("RGBA")
    for t in snapshot.get("tracks",[]):
        mask=t.get("mask")
        if mask is not None:
            layer=np.zeros((*mask.shape,4),np.uint8);layer[mask]=[45,215,144,85]
            rgba=Image.alpha_composite(rgba,Image.fromarray(layer))
    draw=ImageDraw.Draw(rgba)
    candidates=[]
    for q in snapshot.get("queries",[]):
        if q["normalized_query"]==snapshot.get("active_query"):
            candidates.extend((c,q["state"]) for c in q["candidates"][:3] if not c["track_id"])
    candidates.extend((t["candidate"],t["state"]) for t in snapshot.get("tracks",[]))
    for c,state in candidates:
        x0,y0,x1,y1=c["box_xyxy"]
        color=(45,215,144) if state=="FOUND" else (245,182,66)
        draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
        text=f'{c["detector_match"]} · {c["score"]:.2f} · {state}'
        box=draw.textbbox((x0,max(0,y0-16)),text)
        draw.rectangle(box,fill=(18,24,30,235));draw.text((x0,max(0,y0-16)),text,fill=color)
    return rgba.convert("RGB")


def save_output(p,driver,metadata,output,ticks):
    from .query_memory import save_query_memory
    frame=driver.current
    s=p.snapshot(driver.now())
    plain={k:v for k,v in s.items() if k!="tracks"}
    plain["tracks"]=[{k:v for k,v in t.items() if k!="mask"} for t in s["tracks"]]
    plain["input"]=metadata
    plain["core_tick_distribution_ms"]=distribution(ticks)
    plain["decode_ms"]=distribution(driver.decode_ms)
    plain["memory"]=save_query_memory(p,output,metadata["world_frame"])
    start=time.perf_counter()
    (output/"perception.json").write_text(json.dumps(plain,indent=2,allow_nan=False)+"\n")
    serialization_ms=(time.perf_counter()-start)*1000
    (output/"serialization.json").write_text(json.dumps({"final_snapshot_write_ms":serialization_ms})+"\n")
    overlay(frame,s).save(output/"query-overlay.png")
    for t in s["tracks"]:
        if t["mask"] is not None:np.save(output/(t["track_id"]+"-mask.npy"),t["mask"])
    return plain


def run_perception_command(args):
    if args.command in ("prepare-reality","benchmark-reality"):
        from .reality_benchmark import prepare_spec,benchmark
        if args.command=="prepare-reality":
            result=prepare_spec(args.nyu_mat,args.output)
        else:
            result=benchmark(args.nyu_mat,args.spec,args.output,args.split,
                tuple(args.backend or ("closed","grounding-dino","owlv2")),args.device,args.manifest)
        print(json.dumps(result,indent=2));return 0
    if args.command=="hardware":
        print(json.dumps(hardware_report(),indent=2));return 0
    if args.command=="capture-reality":
        from .perception_sources import capture_reality
        result=capture_reality(args.output,args.webcam if args.webcam is not None else args.video,args.frames,args.hz,args.width)
        print(f'Imported {len(result["frames"])} RGB frames into {args.output}');return 0
    output=args.output or Path("runs")/("perception-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    if output.exists() and any(output.iterdir()):raise ValueError("Use an empty output directory")
    output.mkdir(parents=True,exist_ok=True)
    driver,meta=load_source(args)
    p=make_pipeline(args,driver,output)
    ticks=[]
    try:
        if args.interactive:
            from .perception_gui import run_window
            run_window(p,driver,meta,args.phrase if args.command=="query-object" else (args.query[0] if args.query else ""),ticks)
        else:
            queries=[args.phrase] if args.command=="query-object" else args.query
            pending=list(queries)
            if pending:p.request(pending.pop(0))
            start=time.monotonic()
            while time.monotonic()-start<(args.timeout if args.command=="query-object" else args.duration):
                frame=driver.poll(p.runtime)
                stamp=time.perf_counter();s=p.tick(frame,driver.now());ticks.append((time.perf_counter()-stamp)*1000)
                result=p.results.get(p.pending_query[1].normalized) if p.pending_query else None
                if result and result.state not in ("PENDING",):
                    if pending:p.request(pending.pop(0))
                    elif args.command=="query-object":break
                time.sleep(.05)
            if args.command=="query-object" and (result is None or result.state=="PENDING"):
                raise RuntimeError("Query timed out; check runtime failures and model-download access")
        result=save_output(p,driver,meta,output,ticks)
        print(json.dumps({"output":str(output.resolve()),"queries":result["queries"],"memory":result["memory"]},indent=2))
    finally:
        p.close();driver.close()
    return 0
