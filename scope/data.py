"""The small sensor contract needed for known-pose RGB-D mapping."""

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or not np.isfinite(
                [self.fx, self.fy, self.cx, self.cy]).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Invalid pinhole camera intrinsics")

    def scaled(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)


@dataclass(frozen=True)
class RgbdFrame:
    """Depth is optical Z in meters; pose maps optical camera coordinates to world.

    Optical X points right, Y down, Z forward. World follows the supplied poses;
    the synthetic fixture uses a right-handed Z-up world.
    T_world_camera is a 4x4 rigid transform. Timestamps are source seconds.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Intrinsics
    T_world_camera: np.ndarray
    timestamp_s: float
    depth_timestamp_s: float
    source: str
    frame_id: str

    def validate(self, max_pair_offset_s: float = 0.05) -> None:
        self.intrinsics.validate()
        h, w = self.intrinsics.height, self.intrinsics.width
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError(f"{self.frame_id}: RGB must be uint8 HxWx3")
        if self.depth_m.shape != (h, w) or not np.issubdtype(self.depth_m.dtype, np.floating):
            raise ValueError(f"{self.frame_id}: depth must be floating HxW in meters")
        T = self.T_world_camera
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(
                T[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"{self.frame_id}: invalid camera transform")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(
                np.linalg.det(R), 1, atol=1e-4):
            raise ValueError(f"{self.frame_id}: camera rotation is not rigid")
        if not np.isfinite([self.timestamp_s, self.depth_timestamp_s]).all():
            raise ValueError(f"{self.frame_id}: invalid timestamp")
        if abs(self.timestamp_s - self.depth_timestamp_s) > max_pair_offset_s:
            raise ValueError(f"{self.frame_id}: RGB/depth timing differs by more than "
                             f"{max_pair_offset_s:.3f} s")


class RgbdSource(Protocol):
    """A source yields measured or simulated known-pose frames in time order."""

    name: str

    def __iter__(self) -> Iterator[RgbdFrame]: ...

    def __len__(self) -> int: ...


def project_depth(frame: RgbdFrame, stride: int = 1,
                  min_depth_m: float = 0.15, max_depth_m: float = 6.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera-frame endpoints, RGB, and optical depths for valid pixels."""
    h, w = frame.depth_m.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = frame.depth_m[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    z, u, v = z[valid], uu[valid], vv[valid]
    points = np.column_stack(((u - frame.intrinsics.cx) * z / frame.intrinsics.fx,
                              (v - frame.intrinsics.cy) * z / frame.intrinsics.fy, z))
    return points.astype(np.float32), frame.rgb[::stride, ::stride][valid], z


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return points @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
