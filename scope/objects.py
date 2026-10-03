"""Frame-local semantic evidence and depth projection into the mapping frame."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import RgbdFrame, camera_to_world


@dataclass(frozen=True)
class ObjectObservation2D:
    observation_id: str
    frame_id: str
    mask: np.ndarray
    class_probabilities: dict[str, float]
    confidence: float
    source: str
    truth_id: str | None = None  # Evaluation only; never used for association.

    @property
    def label(self) -> str:
        return max(self.class_probabilities, key=self.class_probabilities.get)

    @property
    def box_xyxy(self) -> tuple[int, int, int, int]:
        yy, xx = np.nonzero(self.mask)
        if not len(xx):
            raise ValueError(f"{self.observation_id}: empty mask")
        return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)

    def validate(self, frame: RgbdFrame) -> None:
        if self.frame_id != frame.frame_id or self.mask.shape != frame.depth_m.shape:
            raise ValueError(f"{self.observation_id}: mask does not match frame")
        if self.mask.dtype != np.bool_ or not self.mask.any():
            raise ValueError(f"{self.observation_id}: mask must be nonempty boolean image")
        values = np.array(list(self.class_probabilities.values()), dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError(f"{self.observation_id}: invalid class probabilities")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"{self.observation_id}: invalid confidence")


@dataclass(frozen=True)
class ObjectObservation3D:
    detection: ObjectObservation2D
    timestamp_s: float
    camera_position: np.ndarray
    points_world: np.ndarray
    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    covariance: np.ndarray
    valid_depth_fraction: float
    support_pixels: int
    depth_spread_m: float

    @property
    def observation_id(self) -> str:
        return self.detection.observation_id

    @property
    def label(self) -> str:
        return self.detection.label


class ObjectDetector(Protocol):
    name: str

    def detect(self, frame: RgbdFrame) -> list[ObjectObservation2D]: ...


def project_object(frame: RgbdFrame, detection: ObjectObservation2D,
                   *, max_points: int = 4000, min_support: int = 12,
                   max_depth_m: float = 5.0) -> ObjectObservation3D | None:
    """Project valid mask depths; reject isolated extreme depth outliers.

    The returned bounds describe observed surfaces, not the hidden full object.
    """
    detection.validate(frame)
    vv, uu = np.nonzero(detection.mask)
    depth = frame.depth_m[vv, uu]
    good = np.isfinite(depth) & (depth >= .15) & (depth <= max_depth_m)
    valid_fraction = float(good.mean())
    if good.sum() < min_support:
        return None
    vv, uu, depth = vv[good], uu[good], depth[good]
    median = float(np.median(depth))
    mad = float(np.median(np.abs(depth - median)))
    inlier = np.abs(depth - median) <= max(.35, 4.0 * 1.4826 * mad)
    vv, uu, depth = vv[inlier], uu[inlier], depth[inlier]
    if len(depth) < min_support:
        return None
    # Deterministic sampling keeps viewer files and entity evidence bounded.
    if len(depth) > max_points:
        chosen = np.linspace(0, len(depth) - 1, max_points, dtype=int)
        vv, uu, depth = vv[chosen], uu[chosen], depth[chosen]
    K = frame.intrinsics
    camera = np.column_stack(((uu - K.cx) * depth / K.fx,
                              (vv - K.cy) * depth / K.fy, depth))
    points = camera_to_world(camera, frame.T_world_camera).astype(np.float32)
    low, high = np.percentile(points, [2, 98], axis=0)
    center = np.median(points, axis=0)
    spread = float(1.4826 * mad)
    # This is an observation uncertainty proxy; hidden extent is a separate limit.
    scale = max(.01, .015 * median + spread / np.sqrt(len(points)))
    covariance = np.cov(points, rowvar=False) / len(points) + np.eye(3) * scale**2
    covariance *= 1.0 + 2.0 * (1.0 - valid_fraction)
    return ObjectObservation3D(detection, frame.timestamp_s,
                               frame.T_world_camera[:3, 3].copy(), points,
                               center.astype(np.float32), low.astype(np.float32),
                               high.astype(np.float32), covariance.astype(np.float32),
                               valid_fraction, len(points), spread)
