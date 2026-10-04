"""Offline human/pointing demos and explicit runtime controls."""
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .human_pipeline import HumanPipeline,MODULES
from .human_synthetic import HumanCameraSample,HumanScene,SCENARIOS
from .mapping import MapConfig
from .runtime import Health,ModuleConfig,Runtime,distribution


def register_human_commands(commands):
    for name in ("humans","pointing","telemetry"):
        p = commands.add_parser(name,help=f"Run the offline {name} stages")
        p.add_argument("source",choices=("synthetic","episode","tum","ipo"))
        p.add_argument("dataset",nargs="?",type=Path)
        add_human_options(p)
    p = commands.add_parser("benchmark-humans",help="Oracle pointing, occlusion and multi-view evaluation")
    p.add_argument("--output",type=Path,default=Path("runs/m4-benchmark"))
    p.add_argument("--frames",type=int,default=10)
    p = commands.add_parser("benchmark-ipo",help="Estimated fingers vs labeled public pointing directions")
    p.add_argument("dataset",type=Path)
    p.add_argument("--calibration-folder",type=Path,required=True)
    p.add_argument("--hand-model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p = commands.add_parser("spot-sensors",help="Read-only image capability/timing report; no lease or motion")
    p.add_argument("--hostname",required=True)
    p.add_argument("--samples",type=int,default=5)
    p.add_argument("--source",action="append")
    p.add_argument("--output",type=Path)


def add_human_options(p,mapping=False):
    p.add_argument("--camera",action="append",choices=("front","left","right"))
    p.add_argument("--scenario",choices=SCENARIOS,default="complete")
    if not mapping:
        p.add_argument("--frames",type=int,default=24)
        p.add_argument("--width",type=int,default=320)
        p.add_argument("--frame-stride",type=int,default=1)
        p.add_argument("--output",type=Path)
        p.add_argument("--no-viewer",action="store_true")
    keypoints = p.add_mutually_exclusive_group()
    keypoints.add_argument("--oracle-keypoints",action="store_true")
    keypoints.add_argument("--estimated-keypoints",action="store_true")
    p.add_argument("--disable",action="append",default=[],metavar="MODULE",
                   help="Disable a module, camera/front, depth/left, etc.")
    p.add_argument("--rate",action="append",default=[],metavar="MODULE=HZ")
    p.add_argument("--every",action="append",default=[],metavar="MODULE=N")
    p.add_argument("--config",type=Path,help="JSON object of module names -> ModuleConfig fields")
    p.add_argument("--realtime",action="store_true",help="Paced input and nonblocking inference workers")
    p.add_argument("--toggle-demo",action="store_true",help="Disable a camera, then humans, then resume")
    p.add_argument("--start-frame",type=int,default=0,help="First RGB frame index in TUM input")
    p.add_argument("--hand-model",type=Path,help="Optional local MediaPipe gesture bundle for finger geometry")
    p.add_argument("--calibration-folder",type=Path,help="IPO reference folder with supplied 2D/3D object labels")
    p.add_argument("--no-recording",action="store_true",help="Disable Rerun independently")


def configuration(args,cameras):
    known = set(MODULES)|{"entity_projection"}|{
        f"{prefix}/{c}" for prefix in ("camera","depth","pose","semantic") for c in cameras}
    configs = {}
    def put(name,**changes):
        if name not in known:
            raise ValueError(f"Unknown module {name}; choose from {sorted(known)}")
        configs[name] = replace(configs.get(name,ModuleConfig()),**changes)
    if args.config:
        for name,fields in json.loads(args.config.read_text()).items():
            put(name,**fields)
    for name in args.disable:
        put(name,enabled=False)
    for value in args.rate:
        name,hz = value.split("=",1)
        put(name,max_hz=float(hz))
    for value in args.every:
        name,n = value.split("=",1)
        put(name,every_n=int(n))
    if args.no_recording:
        put("rerun",enabled=False)
    return configs


def summary(snapshot):
    def array(value):
        return None if value is None else value.tolist()
    tracks = [{"person_id":t.person_id,"state":t.state,"last_seen_s":t.last_seen_s,
        "source_cameras":t.pose.source_cameras,"confidence":t.pose.confidence,
        "history_length":len(t.history),"observation_ids":t.pose.observation_ids,
        "hand_quality":t.pose.hand_quality,
        "velocity_m_s":{n:array(v) for n,v in t.velocities.items()},
        "joints":{n:{"position_m":array(j.position),"covariance_m2":array(j.covariance),
            "state":str(j.state),"timestamp_s":j.timestamp_s,"confidence":j.confidence,
            "source_cameras":j.source_cameras,"reason":j.reason} for n,j in t.pose.joints.items()}}
        for t in snapshot["tracks"]]
    pointing = [{"person_track_id":h.person_track_id,"state":h.state,"reason":h.reason,
        "origin_m":array(h.origin),"origin_covariance_m2":array(h.origin_covariance),
        "direction":array(h.direction),"angular_uncertainty_rad":h.angular_uncertainty_rad,
        "quality":h.quality,"timestamp_s":h.timestamp_s,"arm":h.arm,
        "source_observations":h.source_observations,"origins_m":array(h.origins),
        "directions":array(h.directions)} for h in snapshot["pointing"]]
    return {"timestamp_s":snapshot["timestamp_s"],"tracks":tracks,"pointing":pointing,
            "candidates":snapshot["candidates"],"unresolved_views":snapshot["unresolved_views"],
            "unresolved_tracks":snapshot["unresolved_tracks"],"memory":snapshot["memory"]}


def run_human_command(args):
    if args.command=="benchmark-ipo":
        from .ipo import benchmark_ipo
        result = benchmark_ipo(args.dataset,args.calibration_folder,args.hand_model,args.output)
        print(json.dumps({k:v for k,v in result.items() if k in (
            "total","nonabstained","angular_error_mean_deg","top1_rate_all","top3_recall_all","abstention_rate")},indent=2))
        return 0
    if args.command=="spot-sensors":
        from .sensor_diagnostic import diagnose_spot
        report = diagnose_spot(args.hostname,args.samples,args.source)
        data = json.dumps(report,indent=2,allow_nan=False)+"\n"
        print(data)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(data)
        return 0
    if args.command=="benchmark-humans":
        if args.frames<1:
            raise ValueError("Positive frame count required")
        from .human_eval import benchmark
        benchmark(args.output,args.frames)
        print(f"Saved benchmark: {args.output.resolve()}")
        return 0
    if args.frames<1 or args.width<64 or args.frame_stride<1:
        raise ValueError("Positive frames/stride and image width at least 64 required")
    if getattr(args,"voxel_m",.1)<=0:
        raise ValueError("Positive voxel size required")
    cameras = tuple(args.camera or (("front","left") if args.source=="synthetic" else ("front",)))
    if args.source!="synthetic" and len(cameras)!=1:
        raise ValueError("Single-camera recordings require one --camera; synthetic supplies calibrated multi-view input")
    if args.oracle_keypoints and args.source!="synthetic":
        raise ValueError("Oracle joints require synthetic truth")
    output = args.output or Path("runs")/f"m4-{args.command}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve prior evidence")
    output.mkdir(parents=True,exist_ok=True)
    configs = configuration(args,cameras)
    if args.command=="humans":
        configs.setdefault("pointing",ModuleConfig(enabled=False))
        configs.setdefault("intersection",ModuleConfig(enabled=False))
        for module in ("mapping","semantic_detection","world_entities","memory"):
            configs.setdefault(module,ModuleConfig(enabled=False))
    if args.command=="map" and not args.entities:
        for module in ("semantic_detection","world_entities","memory"):
            configs.setdefault(module,ModuleConfig(enabled=False))
    if args.source=="synthetic":
        source = HumanScene(args.frames,args.width,cameras,args.scenario)
        batches = iter(source)
        map_config = MapConfig.synthetic()
    elif args.source=="ipo":
        if not args.dataset or not args.calibration_folder:
            raise ValueError("IPO requires a dataset folder and --calibration-folder reference")
        from .ipo import IpoSource
        source = IpoSource(args.dataset,args.calibration_folder,args.frames,cameras[0])
        all_batches = list(source)
        frames = [s[0].frame for s in all_batches]
        if not frames:
            raise ValueError("IPO folder has no RGB-D stills")
        batches = iter(all_batches)
        map_config = MapConfig.around_frames(frames)
        configs.setdefault("memory",ModuleConfig(enabled=False))
    else:
        if not args.dataset:
            raise ValueError("Supply a saved RGB-D episode or extracted TUM directory")
        if args.source=="episode":
            from .storage import load_episode
            frames = load_episode(args.dataset)[:args.frames]
        else:
            from .tum import TumRgbd
            source = TumRgbd(args.dataset,args.frames,args.frame_stride,args.width)
            source.selected = source.rgb[args.start_frame::args.frame_stride][:args.frames]
            frames = list(source)
        if not frames:
            raise ValueError("No synchronized frames with supplied poses")
        batches = iter([[HumanCameraSample(cameras[0],f,[],[])] for f in frames])
        map_config = MapConfig.around_frames(frames)
        # No synthetic semantics or frame alignment claims on real recordings.
        configs.setdefault("memory",ModuleConfig(enabled=False))
    if args.command=="map" and args.voxel_m!=map_config.voxel_m:
        extent = np.asarray(map_config.shape)*map_config.voxel_m
        shape = tuple(int(v) for v in np.ceil(extent/args.voxel_m))
        if np.prod(shape)>2_000_000:
            raise ValueError("Map exceeds 2 million voxels; increase voxel size")
        map_config = replace(map_config,shape=shape,voxel_m=args.voxel_m)
    estimated = args.estimated_keypoints or args.source!="synthetic"
    if args.source!="synthetic" and configs.get("memory",ModuleConfig(enabled=False)).enabled:
        raise ValueError("Recorded memory requires the M3 import command with explicit shared-frame/alignment")
    # Model loading has its own failure boundary; mapping remains runnable.
    rt_init = Runtime()
    pose = None
    if estimated:
        from .pose_detector import TorchvisionPoseDetector
        pose = rt_init.run("pose_initialization",0.,lambda:TorchvisionPoseDetector(hand_model=args.hand_model))
    semantic = None
    if args.source!="synthetic" and configs.get("semantic_detection",ModuleConfig()).enabled:
        from .detector import TorchvisionMaskDetector
        semantic = rt_init.run("semantic_initialization",0.,TorchvisionMaskDetector)
    p = HumanPipeline(cameras,configs,pose,semantic,args.realtime,map_config)
    if estimated and pose is None:
        class FailedPose:
            name = "unavailable-pose-model"
            def detect(self,*unused):
                raise RuntimeError("Pose initialization failed; see initialization telemetry")
        p.pose_detector = FailedPose()
    if args.source!="synthetic" and semantic is None:
        p.runtime.configure("semantic_detection",ModuleConfig(enabled=False))
    if p.runtime.state("robot_visualization").config.enabled:
        p.runtime.mark("robot_visualization",Health.UNAVAILABLE,"No robot model attached to offline perception viewer")
    from .human_viewer import HumanRecorder
    recorder = HumanRecorder(output,cameras)
    records,ages = [],[]
    begin = time.perf_counter()
    snapshot = None
    try:
        index = 0
        while True:
            tick_begin = time.perf_counter()
            # Acquire is timed separately; decode is unavailable for the preloaded
            # episode path. Do not invent a split of input/decode timings.
            holder = []
            def acquire():
                try:
                    holder.append(next(batches))
                except StopIteration:
                    return False
                return True
            has_input = p.runtime.run("acquire_input",index/10.,acquire,source=args.source,scheduled=False)
            if not has_input:
                break
            samples = holder[0]
            now = samples[0].frame.timestamp_s
            acquisition_state = p.runtime.state("acquire_input")
            acquisition_state.last_success_s = acquisition_state.input_timestamp_s = now
            acquisition_state.success_times_s[-1] = now
            if args.toggle_demo:
                if index==args.frames//3 and len(cameras)>1:
                    p.runtime.configure(f"camera/{cameras[-1]}",ModuleConfig(enabled=False))
                if index==args.frames//2:
                    p.runtime.configure("human_pose",ModuleConfig(enabled=False))
                if index==args.frames*3//4:
                    p.runtime.configure("human_pose",ModuleConfig())
                    for c in cameras:
                        p.runtime.configure(f"camera/{c}",ModuleConfig())
            snapshot = p.step(samples,now)
            ages.append((time.perf_counter()-tick_begin)*1000)
            p.runtime.run("visualization",now,lambda:recorder.log(p,snapshot,index))
            records.append(summary(snapshot))
            index += 1
            if args.realtime:
                time.sleep(max(0.,.1-(time.perf_counter()-tick_begin)))
        if snapshot is not None:
            if p.runtime.state("memory").config.enabled:
                p.runtime.run("memory_persist",snapshot["timestamp_s"],lambda:p.save_memory(output),scheduled=False)
            snapshot["memory"] = dict(p.memory_status)
            p.runtime.run("visualization",snapshot["timestamp_s"],lambda:recorder.log(p,snapshot,index),scheduled=False)
        from .storage import save_map
        save_map(output,p.mapping)
        (output/"humans.json").write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
        if pose is not None:
            pose.metadata["hand_telemetry"] = pose.hand_runtime.snapshot(now)
        metrics = {"source":args.source,"scenario":args.scenario if args.source=="synthetic" else None,
            "keypoints":"estimated" if estimated else "oracle","frames":len(records),
            "wall_replay_fps":len(records)/(time.perf_counter()-begin),
            "host_perception_age":distribution(ages),"model":getattr(pose,"metadata",None),
            "initialization":rt_init.snapshot(0.),"modules":p.runtime.snapshot(now),
            "mapping_revisions":p.mapping.revision,"entities":len(p.entities.entities),
            "memory":p.memory_status,"clock":"source seconds; host durations measured separately",
            "realtime":args.realtime}
        if args.source=="ipo":
            metrics["clock"] = source.metadata
            for state in metrics["modules"].values():
                state["source_update_hz"] = None
        (output/"telemetry.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
        (output/"entities.json").write_text(json.dumps(p.entities.summaries(),indent=2)+"\n")
        print(json.dumps({k:metrics[k] for k in ("frames","mapping_revisions","entities","wall_replay_fps","host_perception_age")},indent=2))
        print(f"Saved run: {output.resolve()}")
    finally:
        recorder.close()
        workers_finished = p.close()
        if pose is not None and workers_finished:
            pose.close()
    if not args.no_viewer and (output/"humans.rrd").exists():
        from .viewer import open_recording
        open_recording(output/"humans.rrd")
    return 0
