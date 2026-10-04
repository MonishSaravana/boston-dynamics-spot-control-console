"""Controlled virtual-camera human fixture in the M1/M2 room.

Depth near visible joints is a simulated body surface. RGB skeleton marks are
illustrative and are NOT evidence of neural pose performance.
"""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

from .data import Intrinsics, RgbdFrame
from .humans import BONES, HumanKeypoint2D, PersonObservation2D
from .synthetic import look_at, render, room_boxes

SCENARIOS = ("complete", "wrist_missing", "elbow_missing", "lower_body_outside",
             "partial_frame", "hand_outside", "complementary_views", "noisy_depth",
             "two_people", "overlapping_people", "not_pointing", "outside_cone",
             "no_target", "brief_exit", "camera_loss")


@dataclass
class HumanCameraSample:
    camera_id: str
    frame: RgbdFrame
    observations: list[PersonObservation2D]
    truth: list[dict]


def skeleton(target=None, shift=None, pointing=True):
    shift = np.zeros(3) if shift is None else np.asarray(shift)
    target = (np.array([1.02,.52,.82]) if target is None else np.asarray(target))-shift
    result = {"left_shoulder":np.array([-.55,-.60,1.45]),
              "right_shoulder":np.array([-.19,-.60,1.45]),
              "left_hip":np.array([-.51,-.60,.92]),"right_hip":np.array([-.23,-.60,.92]),
              "nose":np.array([-.37,-.57,1.70]),
              "left_elbow":np.array([-.61,-.60,1.15]),
              "left_wrist":np.array([-.57,-.60,.85]),
              "left_knee":np.array([-.51,-.60,.50]),"right_knee":np.array([-.23,-.60,.50]),
              "left_ankle":np.array([-.51,-.60,.08]),"right_ankle":np.array([-.23,-.60,.08])}
    shoulder = result["right_shoulder"]
    d = target-shoulder
    d /= np.linalg.norm(d)
    if not pointing:
        d = np.array([0.,0.,-1.])
    result["right_elbow"],result["right_wrist"] = shoulder+.30*d,shoulder+.60*d
    return {k:v+shift for k,v in result.items()}


class HumanScene:
    name = "synthetic-human-room-v1"
    def __init__(self,frames=24,width=320,cameras=("front","left"),scenario="complete",distance_m=None):
        if scenario not in SCENARIOS or not cameras or set(cameras)-{"front","left","right"}:
            raise ValueError("Known scenario and at least one virtual camera required")
        self.frames,self.cameras,self.scenario = frames,tuple(cameras),scenario
        self.distance_m = distance_m
        self.boxes = room_boxes()
        self.K = Intrinsics(width,int(width*.75),width*.85,width*.85,(width-1)/2,(int(width*.75)-1)/2)
        eyes = {"front":np.array([-.3,-2.35,1.5]),"left":np.array([-2.25,-.15,1.6]),
                "right":np.array([2.3,-1.6,1.6])}
        self.poses = {c:look_at(eyes[c],np.array([0.,-.1,1.])) for c in cameras}

    def tick(self,index):
        scenario = self.scenario
        target = np.array([1.02,.52,.82])
        if scenario=="no_target":
            target = np.array([1.3,-1.,1.7])
        if scenario=="outside_cone":
            target = np.array([1.7,-.6,1.45])
        shift = np.array([.01*np.sin(index*.2),0,0])
        if self.distance_m is not None:
            v = target-np.array([-.19,-.60,1.45])
            shift += v/np.linalg.norm(v)*(np.linalg.norm(v)-.60-self.distance_m)
        people = [skeleton(target,shift=shift,pointing=scenario!="not_pointing")]
        if scenario in ("two_people","overlapping_people"):
            people.append(skeleton(target,shift=[-.8 if scenario=="two_people" else .04,0,0]))
        if scenario=="brief_exit" and 5<=index<9:
            people = []
        samples = []
        for camera in self.cameras:
            if scenario=="camera_loss" and camera=="left" and index>=self.frames//2:
                continue
            T = self.poses[camera]
            rgb,depth = render(self.boxes,self.K,T)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            observations,truth = [],[]
            timestamp = index/10.
            for person_index,person in enumerate(people):
                keys,visibility,projected = {},{},{}
                for name,p in person.items():
                    cam = T[:3,:3].T@(p-T[:3,3])
                    uv = np.array([self.K.fx*cam[0]/cam[2]+self.K.cx,
                                   self.K.fy*cam[1]/cam[2]+self.K.cy])
                    projected[name] = uv
                    visible = cam[2]>.15 and 3<=uv[0]<self.K.width-3 and 3<=uv[1]<self.K.height-3
                    x,y = np.round(uv).astype(int)
                    if visible:
                        visible = depth[y,x]<=0 or cam[2]<=depth[y,x]+.04
                    if scenario in ("wrist_missing","hand_outside") and name=="right_wrist":
                        visible = False
                    if scenario=="elbow_missing" and name=="right_elbow":
                        visible = False
                    if scenario=="lower_body_outside" and any(s in name for s in ("hip","knee","ankle")):
                        visible = False
                    if scenario=="partial_frame" and name.startswith("left"):
                        visible = False
                    if scenario=="complementary_views" and (
                            camera=="front" and name in ("right_wrist","right_elbow") or
                            camera=="left" and name.startswith("left")):
                        visible = False
                    visibility[name] = bool(visible)
                    if not visible:
                        continue
                    keys[name] = HumanKeypoint2D(name,uv,1.,.6)
                    depth[y-3:y+4,x-3:x+4] = cam[2]
                for a,b in BONES:
                    if a in keys and b in keys:
                        draw.line([tuple(keys[a].uv),tuple(keys[b].uv)],fill=(105,240,178),width=3)
                for k in keys.values():
                    u,v = k.uv
                    draw.ellipse([u-3,v-3,u+3,v+3],fill=(255,245,160))
                if keys:
                    xy = np.array([k.uv for k in keys.values()])
                    observations.append(PersonObservation2D(f"{camera}-{index}:observation-{person_index}",
                        f"{camera}-{index}",camera,timestamp,keys,
                        tuple(np.r_[xy.min(0),xy.max(0)].tolist()),1.,"oracle-visible-joints"))
                truth.append({"person_index":person_index,"keypoints_world":person,
                              "keypoints_2d":projected,"visibility":visibility,
                              "occluded":{n:not v for n,v in visibility.items()},
                              "target":target,"target_identity":"chair_b" if scenario not in (
                                  "no_target","outside_cone","not_pointing") else None,
                              "ray_origin":person["right_wrist"],
                              "ray_direction":(target-person["right_wrist"])/np.linalg.norm(target-person["right_wrist"])})
            if scenario=="noisy_depth":
                rng = np.random.default_rng(index+41)
                depth += rng.normal(0,.04,depth.shape).astype(np.float32)
                depth[rng.random(depth.shape)<.1] = np.nan
            frame = RgbdFrame(np.asarray(image),depth,self.K,T,timestamp,timestamp,"SIMULATED",f"{camera}-{index}")
            samples.append(HumanCameraSample(camera,frame,observations,truth))
        return samples

    def __iter__(self):
        for i in range(self.frames):
            yield self.tick(i)
