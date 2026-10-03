"""Deterministic repeat visits; truth labels stay outside memory matching."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import RgbdFrame
from .mapping import MapConfig, VoxelMap
from .memory_store import MapAlignment, episode_snapshot
from .semantic_pipeline import SemanticRun, run_semantics
from .semantic_storage import save_semantic_run
from .storage import save_episode, save_map
from .synthetic import Box, SyntheticRoom, look_at, render
from .truth import SyntheticTruthDetector, semantic_identity

SCENARIOS = ("unchanged", "moved_backpack", "moved_chair", "removed", "unobserved",
             "new_object", "ambiguous_chairs", "occluded", "missing_detections",
             "class_disagreement", "depth_noise", "alignment_10cm", "alignment_25cm",
             "alignment_1m", "alignment_1m_undeclared")
START_S = datetime(2026, 10, 3, 14, tzinfo=timezone.utc).timestamp()


@dataclass
class Visit:
    episode_id: str
    room: SyntheticRoom
    run: SemanticRun
    alignment: MapAlignment
    truth: dict[str, dict]


def _truth(boxes):
    result = {}
    for part in boxes:
        tag = semantic_identity(part.name)
        if tag is None:
            continue
        identity, label = tag
        item = result.setdefault(identity, {"label": label, "lows": [], "highs": []})
        item["lows"].append(part.low)
        item["highs"].append(part.high)
    return {identity: {"label": item["label"],
                        "center_m": ((np.min(item["lows"], axis=0)+
                                      np.max(item["highs"], axis=0))/2).tolist()}
            for identity, item in result.items()}


def scenario_visits(name: str, frames: int = 8, width: int = 128) -> list[Visit]:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown repeat-visit scenario {name}")
    visits = []
    for index in range(2):
        room = SyntheticRoom(frames, width, width*3//4)
        boxes = []
        for part in room.boxes:
            shift = np.zeros(3)
            if index:
                if name == "removed" and part.name == "backpack":
                    continue
                if name == "moved_backpack" and part.name == "backpack":
                    shift = np.array([-1.05, .12, .84])
                if name == "moved_chair" and part.name.startswith("chair A"):
                    shift = np.array([-.45, -.65, 0.])
                if name == "ambiguous_chairs" and part.name.startswith("chair"):
                    shift = np.array([.68 if part.name.startswith("chair A") else -.68, -.85, 0.])
            boxes.append(replace(part, low=tuple(np.array(part.low)+shift),
                                 high=tuple(np.array(part.high)+shift)))
        if index and name == "new_object":
            boxes.append(Box("new box", (-1.8, .85, 0), (-1.45, 1.2, .40), (130, 90, 170)))
        if index and name == "occluded":
            boxes.append(Box("occluder", (.50, .1, 0), (1.5, .6, 1.7), (100, 110, 130)))
        room.boxes = tuple(boxes)
        episode_id = f"episode_{index+1:03d}"
        source_frames = list(room)
        if index and name in ("unobserved", "occluded"):
            source_frames = []
            K = replace(room.K, fx=width*1.8, fy=width*1.8)
            for n in range(frames):
                eye = np.array([-1.8+.12*n/frames, -1.3, 1.25]) if name == "unobserved" \
                    else np.array([.9+.10*n/frames, -1.5, 1.30])
                target = np.array([-1.15, .15, .65]) if name == "unobserved" else np.array([1., .85, .5])
                pose = look_at(eye, target)
                rgb, depth = render(room.boxes, K, pose)
                source_frames.append(RgbdFrame(rgb, depth, K, pose, n/5, n/5,
                                               "SIMULATED", f"synthetic-{n:04d}"))
        source_frames = [replace(f, timestamp_s=START_S+index*86400+f.timestamp_s,
                                 depth_timestamp_s=START_S+index*86400+f.depth_timestamp_s,
                                 frame_id=f"{episode_id}-{n:04d}")
                         for n, f in enumerate(source_frames)]
        detector = SyntheticTruthDetector(room.boxes)
        groups = {f.frame_id: detector.detect(f) for f in source_frames}
        if index and name == "missing_detections":
            groups = {key:[d for d in group if d.label != "backpack"] for key,group in groups.items()}
        if index and name == "class_disagreement":
            groups = {key:[replace(d, class_probabilities={"chair": .55, "stool": .45})
                           if d.label == "chair" else d for d in group] for key,group in groups.items()}
        if index and name == "depth_noise":
            rng = np.random.default_rng(20261003)
            noisy = []
            for f in source_frames:
                depth = f.depth_m.copy()
                good = depth > 0
                depth[good] += rng.normal(0, .04, good.sum()).astype(np.float32)
                depth[rng.random(depth.shape)<.1] = 0
                noisy.append(replace(f, depth_m=depth))
            source_frames = noisy
        class CachedDetector:
            name = "synthetic-truth"  # Perfect capability except explicitly dropped detections.
            def detect(self, frame):
                return groups[frame.frame_id]
        run = run_semantics(source_frames, CachedDetector())
        transform = np.eye(4)
        sigma = 0.
        if index and name.startswith("alignment_"):
            error = {"alignment_10cm": .1, "alignment_25cm": .25,
                     "alignment_1m": 1., "alignment_1m_undeclared": 1.}[name]
            transform[0, 3] = error
            sigma = 0. if name.endswith("undeclared") else error
        alignment = MapAlignment("synthetic-room", "known_shared" if sigma==0. and
                                 np.allclose(transform, np.eye(4)) else "external_rigid",
                                 transform, sigma)
        visits.append(Visit(episode_id, room, run, alignment, _truth(room.boxes)))
    return visits


def save_visit(visit: Visit, directory: Path) -> dict:
    save_episode(directory, visit.run.frames, visit.room.name)
    save_semantic_run(directory, visit.run)
    mapping = VoxelMap(MapConfig.synthetic())
    for frame in visit.run.frames:
        if not mapping.integrate(frame):
            raise ValueError(mapping.rejected[-1])
    save_map(directory, mapping)
    return episode_snapshot(visit.episode_id, directory, visit.run, visit.alignment,
                            time_domain="unix_utc")
