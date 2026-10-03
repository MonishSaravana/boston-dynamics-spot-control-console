"""Exact synthetic instance masks; this detector sees no RGB model predictions."""

import numpy as np

from .data import RgbdFrame
from .objects import ObjectObservation2D
from .synthetic import Box, cast_boxes


def semantic_identity(part_name: str) -> tuple[str, str] | None:
    if part_name.startswith("chair A"):
        return "chair_a", "chair"
    if part_name.startswith("chair B"):
        return "chair_b", "chair"
    if part_name.startswith("table"):
        return "table", "table"
    if part_name == "backpack":
        return "backpack", "backpack"
    return None


class SyntheticTruthDetector:
    name = "synthetic-truth"

    def __init__(self, boxes: tuple[Box, ...]):
        self.boxes = boxes

    def detect(self, frame: RgbdFrame) -> list[ObjectObservation2D]:
        K = frame.intrinsics
        vv, uu = np.mgrid[:K.height, :K.width]
        rays_camera = np.column_stack(((uu.ravel() - K.cx) / K.fx,
                                       (vv.ravel() - K.cy) / K.fy,
                                       np.ones(uu.size)))
        rays_world = rays_camera @ frame.T_world_camera[:3, :3].T
        origin = np.broadcast_to(frame.T_world_camera[:3, 3], rays_world.shape)
        _, part_ids = cast_boxes(origin, rays_world, self.boxes)
        result = []
        for identity, label in (("chair_a", "chair"), ("chair_b", "chair"),
                                ("table", "table"), ("backpack", "backpack")):
            indices = [i for i, part in enumerate(self.boxes)
                       if semantic_identity(part.name) == (identity, label)]
            mask = np.isin(part_ids, indices).reshape(K.height, K.width)
            if mask.sum() < 12:
                continue
            result.append(ObjectObservation2D(
                f"{frame.frame_id}:{identity}", frame.frame_id, mask,
                {label: 1.0}, 1.0, self.name, identity))
        return result
