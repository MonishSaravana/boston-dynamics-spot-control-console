"""Known-pose reader for the public TUM RGB-D directory format."""

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .data import Intrinsics, RgbdFrame


def _entries(path: Path) -> list[tuple[float, str]]:
    return [(float(parts[0]), parts[1]) for line in path.read_text().splitlines()
            if (parts := line.split()) and not line.startswith("#")]


def _poses(path: Path) -> list[tuple[float, np.ndarray]]:
    result = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 8:
            continue
        t, x, y, z, qx, qy, qz, qw = values
        q = np.array([qx, qy, qz, qw], dtype=float)
        q /= np.linalg.norm(q)
        xq, yq, zq, wq = q
        R = np.array([
            [1-2*(yq*yq+zq*zq), 2*(xq*yq-zq*wq), 2*(xq*zq+yq*wq)],
            [2*(xq*yq+zq*wq), 1-2*(xq*xq+zq*zq), 2*(yq*zq-xq*wq)],
            [2*(xq*zq-yq*wq), 2*(yq*zq+xq*wq), 1-2*(xq*xq+yq*yq)],
        ])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [x, y, z]
        result.append((t, T))
    return result


def _nearest(times: np.ndarray, stamp: float) -> int:
    index = np.searchsorted(times, stamp)
    return min((max(0, index-1), min(len(times)-1, index)),
               key=lambda i: abs(times[i]-stamp))


class TumRgbd:
    """TUM PNG images and groundtruth.txt; poses are supplied, never estimated."""

    name = "tum-rgbd"

    def __init__(self, directory: Path, max_frames: int = 80, frame_stride: int = 5,
                 width: int = 320):
        self.directory = Path(directory)
        self.rgb = _entries(self.directory / "rgb.txt")
        self.depth = _entries(self.directory / "depth.txt")
        self.poses = _poses(self.directory / "groundtruth.txt")
        if not self.depth or not self.poses or not self.rgb:
            raise ValueError("TUM directory has no RGB, depth, or ground-truth entries")
        self.depth_times = np.array([t for t, _ in self.depth])
        self.pose_times = np.array([t for t, _ in self.poses])
        self.selected = self.rgb[::frame_stride][:max_frames]
        self.width = width

    def __len__(self) -> int:
        return len(self.selected)

    def __iter__(self) -> Iterator[RgbdFrame]:
        from PIL.Image import Resampling
        for index, (stamp, rgb_path) in enumerate(self.selected):
            di = _nearest(self.depth_times, stamp)
            pi = _nearest(self.pose_times, stamp)
            depth_stamp, depth_path = self.depth[di]
            pose_stamp, pose = self.poses[pi]
            if abs(depth_stamp - stamp) > .05 or abs(pose_stamp - stamp) > .05:
                continue
            rgb = np.asarray(Image.open(self.directory / rgb_path).convert("RGB"))
            depth_raw = np.asarray(Image.open(self.directory / depth_path))
            if rgb.shape[:2] != depth_raw.shape:
                continue
            h, w = rgb.shape[:2]
            # TUM recommends the undistorted ROS-default intrinsics for its
            # already registered RGB/depth images; depth PNG scale is 5000/m.
            K = Intrinsics(w, h, 525., 525., 319.5, 239.5)
            if self.width != w:
                new_h = round(h * self.width / w)
                rgb = np.asarray(Image.fromarray(rgb).resize(
                    (self.width, new_h), Resampling.BILINEAR))
                depth_raw = np.asarray(Image.fromarray(depth_raw).resize(
                    (self.width, new_h), Resampling.NEAREST))
                K = K.scaled(self.width, new_h)
            depth_m = depth_raw.astype(np.float32) / 5000.
            yield RgbdFrame(rgb, depth_m, K, pose.copy(), stamp, depth_stamp,
                            "MEASURED_TUM", f"tum-{index:04d}")
