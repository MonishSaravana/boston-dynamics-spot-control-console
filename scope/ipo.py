"""Public IPO pointing stills. Annotation-derived calibration is explicit.

The camera defines the world origin; optical coordinates rotate into a Z-up
camera-local frame. There is no claimed room localization or acquisition FPS.
"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .data import Intrinsics,RgbdFrame
from .human_synthetic import HumanCameraSample


def calibration_from_reference(reference_folder):
    data = loadmat(Path(reference_folder)/"objLocGroundTruth.mat")
    uv,xyz = data["obj2DLoc"],data["obj3DLoc"]
    values,residuals = [],[]
    for axis in (0,1):
        A = np.column_stack((xyz[:,axis]/xyz[:,2],np.ones(len(xyz))))
        fit = np.linalg.lstsq(A,uv[:,axis],rcond=None)[0]
        values.append(fit)
        residuals.extend((A@fit-uv[:,axis]).tolist())
    K = Intrinsics(640,480,values[0][0],values[1][0],values[0][1],values[1][1])
    K.validate()
    return K,{"kind":"pinhole fit to reference annotation 2D/3D correspondences",
        "reference":str(Path(reference_folder)/"objLocGroundTruth.mat"),
        "residual_rms_px":float(np.sqrt(np.mean(np.array(residuals)**2))),
        "depth_scale":"PNG millimeters; checked against supplied optical-Z annotations",
        "world_frame":"camera-local Z-up, declared rigid axis conversion",
        "acquisition_timestamp_s":None,"source_update_hz":None,
        "timestamp_note":"Ordered still index used for replay only; capture intervals unavailable"}


class IpoSource:
    def __init__(self,directory,reference_folder,frames=11,camera="front"):
        self.directory = Path(directory)
        self.K,self.metadata = calibration_from_reference(reference_folder)
        self.T = np.eye(4)
        self.T[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
        self.frames,self.camera = frames,camera
        self.metadata["input_failures"] = []

    def __iter__(self):
        paths = sorted(p for p in self.directory.glob("*.png") if p.stem.isdigit())[:self.frames]
        for i,path in enumerate(paths):
            if not path.with_name(path.stem+"d.png").exists():
                self.metadata["input_failures"].append({"image":path.stem,"reason":"Aligned depth file missing"})
                continue
            rgb = np.asarray(Image.open(path).convert("RGB"))
            depth = np.asarray(Image.open(path.with_name(path.stem+"d.png"))).astype(np.float32)/1000.
            f = RgbdFrame(rgb,depth,self.K,self.T.copy(),float(i),float(i),
                          "PUBLIC_IPO_ORDERED_STILLS",f"ipo-{self.directory.name}-{path.stem}")
            f.validate()
            yield [HumanCameraSample(self.camera,f,[],[])]


def benchmark_ipo(directory,reference_folder,hand_model,output):
    from .humans import lift_pose
    from .person_tracking import PersonTracker
    from .pointing import estimate_pointing
    from .pose_detector import TorchvisionPoseDetector
    from .runtime import Runtime
    output.mkdir(parents=True,exist_ok=True)
    source = IpoSource(directory,reference_folder)
    truth = loadmat(Path(directory)/"objLocGroundTruth.mat")
    model = TorchvisionPoseDetector(hand_model=hand_model)
    runtime = Runtime()
    rows = []
    try:
        for i,batch in enumerate(source):
            i = int(batch[0].frame.frame_id.rsplit("-",1)[1])-1
            if i==0:
                continue  # Explicit dataset no-pointing reference, tested separately.
            f = batch[0].frame
            observations = runtime.run("pose_inference",float(i),lambda:model.detect(f,"front"))
            poses = runtime.run("lifting",float(i),lambda:[lift_pose(o,f) for o in observations])
            tracks = runtime.run("tracking",float(i),lambda:PersonTracker().update(poses,float(i)))
            hypotheses = runtime.run("pointing",float(i),lambda:[estimate_pointing(t,float(i)) for t in tracks])
            hand_truth = loadmat(Path(directory)/f"{i+1:05d}handGt.mat")["hand3DLoc"][0]
            object_points = truth["obj3DLoc"]@source.T[:3,:3].T
            origin = source.T[:3,:3]@hand_truth
            direction = object_points[i-1]-origin
            direction /= np.linalg.norm(direction)
            valid = [h for h in hypotheses if h.state!="ABSTAIN"]
            row = {"image":f.frame_id,"target":str(truth["objList"][i-1]).strip(),
                "state":"ABSTAIN","angular_error_deg":None,"target_rank":None,
                "angular_spread_deg":None,"target_distance_m":float(np.linalg.norm(object_points[i-1]-origin))}
            if len(valid)==1:
                h = valid[0]
                vectors = object_points-h.origin
                dirs = vectors/np.linalg.norm(vectors,axis=1)[:,None]
                angles = np.arccos(np.clip(dirs@h.direction,-1,1))
                row.update({"state":h.state,"angular_error_deg":float(np.rad2deg(np.arccos(np.clip(h.direction@direction,-1,1)))),
                    "target_rank":int(np.where(np.argsort(angles)==i-1)[0][0]+1),
                    "angular_spread_deg":float(np.rad2deg(h.angular_uncertainty_rad)),
                    "origin_error_from_hand_center_m":float(np.linalg.norm(h.origin-origin)),
                    "reason":h.reason})
            rows.append(row)
            print(f"{row['image']}: {row['state']} angle={row['angular_error_deg']} rank={row['target_rank']}",flush=True)
    finally:
        model.close()
    for failure in source.metadata["input_failures"]:
        index = int(failure["image"])-2
        if index>=0:
            rows.append({"image":failure["image"],"target":str(truth["objList"][index]).strip(),
                "state":"UNAVAILABLE","reason":failure["reason"],"angular_error_deg":None,
                "target_rank":None,"angular_spread_deg":None})
    valid = [r for r in rows if r["angular_error_deg"] is not None]
    result = {"source":"public IPO natural-pointing stills", "directory":str(directory),
        "calibration":source.metadata,"samples":rows,"total":len(rows),"nonabstained":len(valid),
        "angular_error_mean_deg":float(np.mean([r["angular_error_deg"] for r in valid])) if valid else None,
        "top1_rate_all":sum(r["target_rank"]==1 for r in rows)/len(rows),
        "top3_recall_all":sum(r["target_rank"] is not None and r["target_rank"]<=3 for r in rows)/len(rows),
        "abstention_rate":1.-len(valid)/len(rows),
        "ranking_kind":"angular ranking of supplied object centers; not neural object detection accuracy",
        "telemetry":runtime.snapshot(10.),"model":model.metadata}
    result["model"]["hand_telemetry"] = model.hand_runtime.snapshot(10.)
    for state in result["telemetry"].values():
        state["source_update_hz"] = None
    (output/"ipo_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result
