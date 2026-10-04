"""Known-pose RGB-D fusion with explicit occupied, free, and unknown voxels."""

from dataclasses import dataclass

import numpy as np

from .data import RgbdFrame, camera_to_world, project_depth


UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


@dataclass(frozen=True)
class MapConfig:
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    voxel_m: float = 0.10
    sample_stride: int = 3
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0

    @classmethod
    def synthetic(cls) -> "MapConfig":
        return cls((-2.8, -2.8, -0.2), (56, 56, 30))

    @classmethod
    def around_frames(cls, frames: list[RgbdFrame], voxel_m: float = 0.10) -> "MapConfig":
        positions = []
        for frame in frames:
            points, _, _ = project_depth(frame, stride=16, max_depth_m=5.)
            if len(points):
                positions.append(camera_to_world(points, frame.T_world_camera))
            positions.append(frame.T_world_camera[:3, 3][None, :])
        xyz = np.concatenate(positions)
        low = np.floor((xyz.min(axis=0) - .4) / voxel_m) * voxel_m
        high = np.ceil((xyz.max(axis=0) + .4) / voxel_m) * voxel_m
        shape = tuple(int(v) for v in np.ceil((high - low) / voxel_m))
        if np.prod(shape) > 2_000_000:
            raise ValueError("Map exceeds 2 million voxels; increase voxel size or limit frames")
        return cls(tuple(low), shape, voxel_m)


class VoxelMap:
    """Ray endpoints mark occupied; traversed cells mark free; untouched stay unknown."""

    def __init__(self, config: MapConfig):
        self.config = config
        self.hits = np.zeros(config.shape, dtype=np.uint16)
        self.frees = np.zeros(config.shape, dtype=np.uint16)
        self.point_sum = np.zeros(config.shape + (3,), dtype=np.float32)
        self.color_sum = np.zeros(config.shape + (3,), dtype=np.float32)
        self.revision = 0
        self.rejected: list[str] = []
        self.warnings: list[str] = []
        self._intrinsics = None
        self._last_timestamp = None
        self._streams = {}

    def _indices(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ijk = np.floor((xyz - self.config.origin) / self.config.voxel_m).astype(np.int32)
        valid = ((ijk >= 0) & (ijk < self.config.shape)).all(axis=1)
        return ijk, valid

    def integrate(self, frame: RgbdFrame, stream_id: str = "default") -> bool:
        try:
            frame.validate()
        except ValueError as exc:
            self.rejected.append(str(exc))
            return False
        intrinsics,last_timestamp = self._streams.get(stream_id,(None,None))
        if intrinsics is not None and frame.intrinsics != intrinsics:
            self.rejected.append(f"{frame.frame_id}: camera calibration changed within episode")
            return False
        if last_timestamp is not None and frame.timestamp_s <= last_timestamp:
            self.rejected.append(f"{frame.frame_id}: timestamp is not increasing")
            return False
        points_cam, colors, depth = project_depth(
            frame, self.config.sample_stride, self.config.min_depth_m,
            self.config.max_depth_m)
        if not len(points_cam):
            self.rejected.append(f"{frame.frame_id}: no valid depth")
            return False
        endpoints = camera_to_world(points_cam, frame.T_world_camera)
        ijk, valid = self._indices(endpoints)
        ijk, endpoints, colors = ijk[valid], endpoints[valid], colors[valid]
        if len(ijk) == 0:
            self.rejected.append(f"{frame.frame_id}: points outside map bounds")
            return False
        if self.revision >= 2:
            # A low overlap score is diagnostic, not an automatic rejection:
            # a genuine new viewpoint may also reveal mostly unseen surfaces.
            from scipy.spatial import cKDTree
            old, _, _ = self.cloud()
            if len(old) > 100:
                sample = endpoints[::max(1, len(endpoints) // 500)]
                nearest, _ = cKDTree(old).query(sample, workers=1)
                overlap = float(np.mean(nearest < .18))
                if overlap < .12:
                    self.warnings.append(
                        f"{frame.frame_id}: low cross-view surface overlap ({overlap:.1%}); "
                        "check pose, calibration, or new-scene motion")
        flat = np.ravel_multi_index(ijk.T, self.config.shape)
        np.add.at(self.hits.ravel(), flat, 1)
        np.add.at(self.point_sum.reshape(-1, 3), flat, endpoints)
        np.add.at(self.color_sum.reshape(-1, 3), flat, colors)

        # Sample along each measured ray, stopping before its surface endpoint.
        # A depth hole contributes neither free space nor a surface.
        valid_depth = depth[valid]
        ray_end = (valid_depth - self.config.voxel_m * .75).clip(min=0)
        steps = np.arange(self.config.voxel_m * .5, self.config.max_depth_m,
                          self.config.voxel_m * .5, dtype=np.float32)
        mask = steps[None, :] < ray_end[:, None]
        fractions = steps[None, :, None] / valid_depth[:, None, None]
        origin = frame.T_world_camera[:3, 3]
        samples = (origin + (endpoints - origin)[:, None, :] * fractions)[mask]
        free_ijk, inside = self._indices(samples)
        if inside.any():
            free_flat = np.ravel_multi_index(free_ijk[inside].T, self.config.shape)
            np.add.at(self.frees.ravel(), free_flat, 1)
        self.revision += 1
        self._intrinsics = frame.intrinsics
        self._last_timestamp = frame.timestamp_s
        self._streams[stream_id] = (frame.intrinsics,frame.timestamp_s)
        return True

    def states(self) -> np.ndarray:
        state = np.zeros(self.config.shape, dtype=np.uint8)
        state[self.frees > 0] = FREE
        state[self.hits > 0] = OCCUPIED
        return state

    def cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.hits > 0
        count = self.hits[mask].astype(np.float32)
        return (self.point_sum[mask] / count[:, None],
                np.clip(self.color_sum[mask] / count[:, None], 0, 255).astype(np.uint8),
                count)

    def topdown(self, min_z: float = .25, max_z: float = 1.5) -> np.ndarray:
        z = np.arange(self.config.shape[2]) * self.config.voxel_m + self.config.origin[2]
        section = self.states()[:, :, (z >= min_z) & (z <= max_z)]
        result = np.zeros(section.shape[:2], dtype=np.uint8)
        result[np.any(section == FREE, axis=2)] = FREE
        result[np.any(section == OCCUPIED, axis=2)] = OCCUPIED
        return result


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: np.ndarray
    triangles: np.ndarray


class DenseTsdf:
    """Known-pose projective TSDF with measured support and bounded memory.

    This deliberately uses a dense grid for a room-scale milestone. Unknown
    voxels have zero weight and are masked out of mesh extraction.
    """

    def __init__(self, config: MapConfig, truncation_m: float = .20):
        if np.prod(config.shape) > 2_000_000:
            raise ValueError("Dense TSDF is limited to 2 million voxels")
        self.config = config
        self.truncation_m = truncation_m
        self.value = np.ones(config.shape, dtype=np.float32)
        self.weight = np.zeros(config.shape, dtype=np.uint16)
        self.integrated = 0
        ijk = np.indices(config.shape, dtype=np.float32).reshape(3, -1).T
        self.centers = np.asarray(config.origin) + (ijk + .5) * config.voxel_m

    def integrate(self, frame: RgbdFrame, max_depth_m: float = 5.) -> None:
        K = frame.intrinsics
        R = frame.T_world_camera[:3, :3]
        t = frame.T_world_camera[:3, 3]
        flat_value, flat_weight = self.value.ravel(), self.weight.ravel()
        for start in range(0, len(self.centers), 100_000):
            stop = min(start + 100_000, len(self.centers))
            xyz = (self.centers[start:stop] - t) @ R
            z = xyz[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.rint(K.fx * xyz[:, 0] / z + K.cx).astype(np.int32)
                v = np.rint(K.fy * xyz[:, 1] / z + K.cy).astype(np.int32)
            in_view = ((z > .15) & (z <= max_depth_m) & (u >= 0) & (u < K.width) &
                       (v >= 0) & (v < K.height))
            local = np.flatnonzero(in_view)
            if not len(local):
                continue
            d = frame.depth_m[v[local], u[local]]
            sdf = d - z[local]
            valid = (np.isfinite(d) & (d >= .15) & (d <= max_depth_m) &
                     (sdf >= -self.truncation_m))
            ids = local[valid] + start
            if not len(ids):
                continue
            observed = np.clip(sdf[valid] / self.truncation_m, -1, 1)
            old_weight = flat_weight[ids].astype(np.float32)
            flat_value[ids] = (flat_value[ids] * old_weight + observed) / (old_weight + 1)
            flat_weight[ids] = np.minimum(flat_weight[ids].astype(np.uint32) + 1, 65535)
        self.integrated += 1

    def mesh(self) -> SurfaceMesh:
        from skimage.measure import marching_cubes
        if not np.any(self.value[self.weight > 0] < 0):
            return SurfaceMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
        vertices, triangles, _, _ = marching_cubes(
            self.value, level=0., spacing=(self.config.voxel_m,) * 3,
            mask=self.weight > 0)
        vertices += np.asarray(self.config.origin) + self.config.voxel_m * .5
        return SurfaceMesh(vertices, triangles.astype(np.int32))
