"""Deterministic room, exact box ray casting, and known camera trajectory."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .data import Intrinsics, RgbdFrame


@dataclass(frozen=True)
class Box:
    name: str
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    color: tuple[int, int, int]


def room_boxes() -> tuple[Box, ...]:
    boxes = [
        Box("floor", (-2.55, -2.55, -0.10), (2.55, 2.55, 0), (125, 132, 139)),
        Box("north wall", (-2.55, 2.48, 0), (2.55, 2.56, 2.65), (199, 208, 216)),
        Box("south wall", (-2.55, -2.56, 0), (2.55, -2.48, 2.65), (205, 211, 214)),
        Box("west wall", (-2.56, -2.55, 0), (-2.48, 2.55, 2.65), (179, 194, 204)),
        Box("east wall", (2.48, -2.55, 0), (2.56, 2.55, 2.65), (179, 194, 204)),
        Box("table top", (-0.45, 0.55, 0.76), (0.55, 1.40, 0.84), (160, 104, 63)),
        Box("table left leg", (-0.42, 0.58, 0), (-0.35, 0.65, 0.77), (116, 76, 50)),
        Box("table right leg", (0.43, 0.58, 0), (0.50, 0.65, 0.77), (116, 76, 50)),
        Box("chair A seat", (-1.30, -0.12, 0.42), (-0.70, 0.45, 0.50), (55, 145, 187)),
        Box("chair A back", (-1.30, 0.39, 0.42), (-0.70, 0.47, 1.16), (43, 115, 160)),
        Box("chair B seat", (0.72, -0.02, 0.42), (1.32, 0.55, 0.50), (221, 155, 58)),
        Box("chair B back", (0.72, 0.49, 0.42), (1.32, 0.57, 1.16), (188, 113, 44)),
        Box("backpack", (0.81, 0.69, 0), (1.20, 0.99, 0.48), (164, 62, 90)),
    ]
    for x in (-1.25, -0.77):
        for y in (-0.08, 0.39):
            boxes.append(Box("chair A leg", (x, y, 0), (x + .06, y + .06, .43),
                             (35, 96, 135)))
    for x in (.76, 1.24):
        for y in (.02, .49):
            boxes.append(Box("chair B leg", (x, y, 0), (x + .06, y + .06, .43),
                             (149, 89, 36)))
    return tuple(boxes)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((right, down, forward))
    T[:3, 3] = eye
    return T


def cast_boxes(origins: np.ndarray, directions: np.ndarray,
               boxes: tuple[Box, ...], max_depth_m: float = 6.) -> tuple[np.ndarray, np.ndarray]:
    """Ray-box intersections. Direction Z need not be unit length; t is optical depth."""
    depth = np.full(len(directions), np.inf, dtype=np.float64)
    ids = np.full(len(directions), -1, dtype=np.int16)
    for box_id, box in enumerate(boxes):
        lo, hi = np.asarray(box.low), np.asarray(box.high)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lo - origins) / directions
            b = (hi - origins) / directions
        near = np.max(np.minimum(a, b), axis=1)
        far = np.min(np.maximum(a, b), axis=1)
        t = np.where(near > 1e-5, near, far)
        valid = (far >= np.maximum(near, 0)) & (t > 0.15) & (t < depth)
        depth[valid], ids[valid] = t[valid], box_id
    depth[depth > max_depth_m] = 0
    ids[depth == 0] = -1
    return depth.astype(np.float32), ids


def render(boxes: tuple[Box, ...], K: Intrinsics, T_world_camera: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    vv, uu = np.mgrid[:K.height, :K.width]
    rays_camera = np.column_stack(((uu.ravel() - K.cx) / K.fx,
                                   (vv.ravel() - K.cy) / K.fy,
                                   np.ones(uu.size)))
    rays_world = rays_camera @ T_world_camera[:3, :3].T
    origins = np.broadcast_to(T_world_camera[:3, 3], rays_world.shape)
    depth, ids = cast_boxes(origins, rays_world, boxes)
    rgb = np.full((uu.size, 3), (35, 43, 57), dtype=np.uint8)
    for i, box in enumerate(boxes):
        rgb[ids == i] = box.color
    return rgb.reshape(K.height, K.width, 3), depth.reshape(K.height, K.width)


class SyntheticRoom:
    name = "synthetic-room-v1"

    def __init__(self, frames: int = 28, width: int = 192, height: int = 144):
        self.frames = frames
        self.K = Intrinsics(width, height, width * 0.92, height * 1.23,
                            (width - 1) / 2, (height - 1) / 2)
        self.boxes = room_boxes()

    def __len__(self) -> int:
        return self.frames

    def __iter__(self) -> Iterator[RgbdFrame]:
        for i in range(self.frames):
            phase = i / max(1, self.frames - 1)
            eye = np.array([-1.65 + 3.3 * phase,
                            -1.48 + .28 * np.sin(phase * np.pi), 1.35])
            target = np.array([0.0, 0.35, 0.83])
            T = look_at(eye, target)
            rgb, depth = render(self.boxes, self.K, T)
            yield RgbdFrame(rgb, depth, self.K, T, i / 5, i / 5,
                            "SIMULATED", f"synthetic-{i:04d}")


def geometry_occupied(points: np.ndarray, boxes: tuple[Box, ...]) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for box in boxes:
        result |= np.all((points >= box.low) & (points <= box.high), axis=1)
    return result
