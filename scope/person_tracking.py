"""Conservative geometric multi-view association and short-lived person tracks."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .humans import HumanJoint3D, HumanPoseObservation, JointState

ANCHORS = ("left_shoulder","right_shoulder","left_hip","right_hip")


def compatibility(a, b):
    if abs(a.timestamp_s-b.timestamp_s)>.08:
        return None
    shared = [name for name in ANCHORS if a.joints[name].position is not None
              and b.joints[name].position is not None]
    if not shared:
        return None
    distances = [np.linalg.norm(a.joints[n].position-b.joints[n].position) for n in shared]
    # One common anchor is permitted only inside a very tight spatial gate.
    if max(distances) > (.10 if len(shared)==1 else .30):
        return None
    return float(np.mean(distances))


def fuse_group(group):
    joints = {}
    timestamp = max(p.timestamp_s for p in group)
    for name in group[0].joints:
        available = [p.joints[name] for p in group if p.joints[name].position is not None]
        if not available:
            joints[name] = group[0].joints[name]
            continue
        positions = np.array([j.position for j in available])
        if len(available)>1 and np.max(np.linalg.norm(positions-positions.mean(0),axis=1))>.20:
            joints[name] = HumanJoint3D(name,None,None,0.,JointState.UNAVAILABLE,timestamp,(),
                                        "Conflicting camera joint observations")
            continue
        information = [np.linalg.inv(j.covariance+np.eye(3)*1e-6) for j in available]
        cov = np.linalg.inv(sum(information))
        point = cov @ sum(I@j.position for I,j in zip(information,available))
        # Disagreement contributes uncertainty, rather than falsely improving it.
        residual = positions-point
        cov += residual.T@residual/len(available)
        cameras = tuple(sorted(set(c for j in available for c in j.source_cameras)))
        joints[name] = HumanJoint3D(name,point,cov,max(j.confidence for j in available),
            JointState.OBSERVED_3D,timestamp,cameras,"Fused aligned depth")
    return HumanPoseObservation(tuple(i for p in group for i in p.observation_ids),timestamp,
        joints,tuple(sorted(set(c for p in group for c in p.source_cameras))),
        float(np.mean([p.confidence for p in group])))


def fuse_views(observations):
    """Mutual-unique matching. Ambiguous detections remain separate and unresolved.

    Association never consumes oracle person IDs or appearance identity labels.
    At most one detection per camera can contribute to a group at a timestamp.
    """
    groups, unresolved = [], []
    cameras = sorted(set(c for p in observations for c in p.source_cameras))
    for camera in cameras:
        found = [p for p in observations if p.source_cameras==(camera,)]
        if not groups:
            groups.extend([[p] for p in found])
            continue
        costs = np.full((len(found),len(groups)), np.inf)
        for i,p in enumerate(found):
            for j,g in enumerate(groups):
                if camera not in fuse_group(g).source_cameras:
                    score = compatibility(p,fuse_group(g))
                    if score is not None:
                        costs[i,j] = score
        assignments = {}
        for i,p in enumerate(found):
            order = np.argsort(costs[i])
            if not len(order) or not np.isfinite(costs[i,order[0]]):
                continue
            j = order[0]
            competitors = np.sort(costs[:,j])
            if (len(order)>1 and costs[i,order[1]]-costs[i,j]<.08 or
                    len(competitors)>1 and competitors[1]-competitors[0]<.08):
                unresolved.append(p.observation_ids)
                continue
            if i==np.argmin(costs[:,j]):
                assignments[i] = j
        for i,p in enumerate(found):
            if i in assignments:
                groups[assignments[i]].append(p)
            else:
                groups.append([p])
    return [fuse_group(g) for g in groups], unresolved


@dataclass
class PersonTrack:
    person_id: str
    pose: HumanPoseObservation
    last_seen_s: float
    state: str = "VISIBLE"
    velocities: dict = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=30))


class PersonTracker:
    def __init__(self, retention_s=1., joint_hold_s=.20):
        self.retention_s, self.joint_hold_s = retention_s,joint_hold_s
        self.tracks = {}
        self.next_id = 1
        self.unresolved = []

    def update(self, poses, timestamp_s):
        self.expire(timestamp_s)
        self.unresolved = []
        used = set()
        existing = list(self.tracks.values())
        candidates = {}
        for i,p in enumerate(poses):
            scores = []
            for t in existing:
                # Compare against the last pose in the current source clock.
                previous = HumanPoseObservation(t.pose.observation_ids,p.timestamp_s,
                    t.pose.joints,t.pose.source_cameras,t.pose.confidence)
                distance = compatibility(p,previous)
                if distance is not None:
                    scores.append((distance,t.person_id))
            candidates[i] = sorted(scores)
        for i,p in enumerate(poses):
            scores = candidates[i]
            ambiguous = len(scores)>1 and scores[1][0]-scores[0][0]<.08
            if scores and not ambiguous:
                chosen = scores[0][1]
                rivals = [s[0][0] for k,s in candidates.items() if k!=i and s and s[0][1]==chosen]
                ambiguous = bool(rivals and min(rivals)-scores[0][0]<.08)
            if ambiguous or scores and scores[0][1] in used:
                self.unresolved.append(p.observation_ids)
                continue
            if not scores:
                # No torso anchor means no defensible person association.
                if not any(p.joints[n].position is not None for n in ANCHORS):
                    self.unresolved.append(p.observation_ids)
                    continue
                name = f"person_{self.next_id:03d}"
                self.next_id += 1
                track = PersonTrack(name,p,timestamp_s)
                self.tracks[name] = track
            else:
                track = self.tracks[scores[0][1]]
                dt = timestamp_s-track.last_seen_s
                joints = {}
                for name,new in p.joints.items():
                    old = track.pose.joints[name]
                    if new.position is not None and old.position is not None and dt>0:
                        Q = np.eye(3)*(.08*dt)**2
                        prior = old.covariance+Q
                        gain = prior@np.linalg.inv(prior+new.covariance)
                        point = old.position+gain@(new.position-old.position)
                        cov = (np.eye(3)-gain)@prior+np.eye(3)*1e-6
                        joints[name] = HumanJoint3D(name,point,cov,new.confidence,new.state,
                            new.timestamp_s,new.source_cameras,"Temporally filtered depth")
                        track.velocities[name] = (point-old.position)/dt
                    elif new.position is None and old.position is not None and (
                            timestamp_s-old.timestamp_s<=self.joint_hold_s):
                        joints[name] = HumanJoint3D(name,old.position.copy(),old.covariance+
                            np.eye(3)*(.15*(timestamp_s-old.timestamp_s))**2,
                            old.confidence*.6,JointState.ESTIMATED,old.timestamp_s,
                            old.source_cameras,"Brief temporal hold; age retained")
                    else:
                        joints[name] = new
                track.pose = HumanPoseObservation(p.observation_ids,p.timestamp_s,joints,
                    p.source_cameras,p.confidence)
                track.last_seen_s = timestamp_s
            track.state = "VISIBLE"
            track.history.append((timestamp_s,p.observation_ids))
            used.add(track.person_id)
        for track in self.tracks.values():
            if track.person_id not in used:
                track.state = "UNRESOLVED" if self.unresolved else "NOT_OBSERVED"
        return list(self.tracks.values())

    def expire(self, timestamp_s):
        self.tracks = {k:t for k,t in self.tracks.items()
                       if timestamp_s-t.last_seen_s<=self.retention_s}
        for t in self.tracks.values():
            if timestamp_s-t.last_seen_s>.25:
                t.state = "STALE"
