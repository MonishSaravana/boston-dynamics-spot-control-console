"""Typed camera observations and robust depth lifting. No model or robot imports."""

from dataclasses import dataclass,field
from enum import StrEnum
from typing import Protocol

import numpy as np

from .data import RgbdFrame

JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle")
HAND_JOINTS = tuple(f"{s}_index_{n}" for s in ("left","right") for n in ("mcp","pip","tip"))
BONES = tuple((f"{side}_{a}", f"{side}_{b}") for side in ("left", "right")
              for a,b in (("shoulder","elbow"), ("elbow","wrist"),
                          ("shoulder","hip"), ("hip","knee"), ("knee","ankle"))) + (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
BONES += tuple((f"{side}_{a}",f"{side}_{b}") for side in ("left","right")
               for a,b in (("wrist","index_mcp"),("index_mcp","index_pip"),("index_pip","index_tip")))


class JointState(StrEnum):
    OBSERVED_3D = "OBSERVED_3D"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class HumanKeypoint2D:
    name: str
    uv: np.ndarray
    confidence: float  # Heuristic quality, never a calibrated probability.
    sigma_px: float = 2.


@dataclass(frozen=True)
class PersonObservation2D:
    observation_id: str
    frame_id: str
    camera_id: str
    timestamp_s: float
    keypoints: dict[str, HumanKeypoint2D]
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    detector: str
    hand_quality: dict[str,float] = field(default_factory=dict)

    def validate(self, frame):
        if self.frame_id != frame.frame_id or abs(self.timestamp_s-frame.timestamp_s)>1e-6:
            raise ValueError("Pose observation does not belong to this frame")
        if not self.camera_id or not np.isfinite(self.box_xyxy).all():
            raise ValueError("Pose requires a camera ID and finite bounds")
        for name,k in self.keypoints.items():
            if name != k.name or k.uv.shape != (2,) or not np.isfinite(k.uv).all() or not (
                    0 <= k.confidence <= 1) or not np.isfinite(k.sigma_px) or k.sigma_px <= 0:
                raise ValueError("Invalid human keypoint")


@dataclass(frozen=True)
class HumanJoint3D:
    name: str
    position: np.ndarray | None
    covariance: np.ndarray | None
    confidence: float
    state: JointState
    timestamp_s: float
    source_cameras: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class HumanPoseObservation:
    observation_ids: tuple[str, ...]
    timestamp_s: float
    joints: dict[str, HumanJoint3D]
    source_cameras: tuple[str, ...]
    confidence: float
    hand_quality: dict[str,float] = field(default_factory=dict)


class HumanPoseDetector(Protocol):
    name: str
    def detect(self, frame: RgbdFrame, camera_id: str) -> list[PersonObservation2D]: ...


def lift_pose(observation: PersonObservation2D, frame: RgbdFrame,
              min_confidence=.35, radius=2) -> HumanPoseObservation:
    frame.validate()
    observation.validate(frame)
    K, R = frame.intrinsics, frame.T_world_camera[:3,:3]
    joints = {}
    for name in JOINTS+HAND_JOINTS:
        keypoint = observation.keypoints.get(name)
        reason = "Joint not detected"
        point = covariance = None
        quality = 0.
        if keypoint is not None and keypoint.confidence >= min_confidence:
            u,v = keypoint.uv
            if 0<=u<K.width and 0<=v<K.height:
                x,y = int(round(u)), int(round(v))
                x,y = min(x,K.width-1), min(y,K.height-1)
                patch = frame.depth_m[max(0,y-radius):min(K.height,y+radius+1),
                                      max(0,x-radius):min(K.width,x+radius+1)]
                values = patch[np.isfinite(patch)&(patch>=.15)&(patch<=6.)]
                if len(values)>=3:
                    z = float(np.median(values))
                    mad = float(np.median(abs(values-z)))
                    center = frame.depth_m[y,x]
                    # Background discontinuities cannot become exact body geometry.
                    if mad <= .12 and np.isfinite(center) and .15<=center<=6 and abs(center-z)<=max(.10,3*mad):
                        sigma_z = max(.012, 1.4826*mad)
                        ray = np.array([(u-K.cx)/K.fx, (v-K.cy)/K.fy, 1.])
                        point = R @ (ray*z) + frame.T_world_camera[:3,3]
                        J = np.array([[z/K.fx,0,ray[0]], [0,z/K.fy,ray[1]], [0,0,1.]])
                        covariance = R @ J @ np.diag([keypoint.sigma_px**2,
                            keypoint.sigma_px**2, sigma_z**2]) @ J.T @ R.T
                        quality = keypoint.confidence * min(1.,len(values)/patch.size)
                    else:
                        reason = "Depth discontinuity or outlier"
                else:
                    reason = "Insufficient valid aligned depth"
            else:
                reason = "Joint outside image"
        elif keypoint is not None:
            reason = "Low joint confidence"
        joints[name] = HumanJoint3D(name, point, covariance, quality,
            JointState.OBSERVED_3D if point is not None else JointState.UNAVAILABLE,
            frame.timestamp_s, (observation.camera_id,) if point is not None else (),
            "Depth observed" if point is not None else reason)
    # Loose anatomical plausibility gate; reject the distal measurement, do not
    # snap it to a fake location. This cannot resolve same-depth occlusions.
    for side in ("left","right"):
        for a,b in (("shoulder","elbow"),("elbow","wrist")):
            parent,child = joints[f"{side}_{a}"],joints[f"{side}_{b}"]
            if parent.position is not None and child.position is not None and not (
                    .08 <= np.linalg.norm(parent.position-child.position) <= .65):
                joints[child.name] = HumanJoint3D(child.name,None,None,0.,JointState.UNAVAILABLE,
                    child.timestamp_s,(),"Implausible arm segment; depth/pose outlier")
    return HumanPoseObservation((observation.observation_id,),frame.timestamp_s,joints,
                                (observation.camera_id,),observation.confidence,observation.hand_quality)
