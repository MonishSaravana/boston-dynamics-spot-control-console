"""Portable map and observation snapshots; no private data enters Git."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .data import Intrinsics, RgbdFrame
from .mapping import DenseTsdf, MapConfig, SurfaceMesh, VoxelMap


def save_episode(directory: Path, frames: list[RgbdFrame], source_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "episode.npz",
                        rgb=np.stack([f.rgb for f in frames]),
                        depth_m=np.stack([f.depth_m for f in frames]),
                        poses=np.stack([f.T_world_camera for f in frames]),
                        rgb_times=np.array([f.timestamp_s for f in frames]),
                        depth_times=np.array([f.depth_timestamp_s for f in frames]))
    manifest = {"format_version": 1, "source": source_name,
                "origin_kind": frames[0].source,
                "frame_ids": [frame.frame_id for frame in frames],
                "frames": len(frames), "intrinsics": asdict(frames[0].intrinsics),
                "coordinate_convention": "world follows supplied poses (synthetic Z up); optical X right Y down Z forward",
                "pose_convention": "T_world_camera"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_episode(directory: Path) -> list[RgbdFrame]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported episode version")
    K = Intrinsics(**manifest["intrinsics"])
    with np.load(directory / "episode.npz") as data:
        return [RgbdFrame(data["rgb"][i].copy(), data["depth_m"][i].copy(), K,
                          data["poses"][i].copy(), float(data["rgb_times"][i]),
                          float(data["depth_times"][i]), manifest["origin_kind"],
                          manifest.get("frame_ids", [f"replay-{j:04d}" for j in
                                                     range(manifest["frames"])])[i])
                for i in range(manifest["frames"])]


def save_map(directory: Path, mapping: VoxelMap) -> None:
    config = asdict(mapping.config)
    np.savez_compressed(directory / "map.npz", hits=mapping.hits, frees=mapping.frees,
                        point_sum=mapping.point_sum, color_sum=mapping.color_sum,
                        revision=np.array(mapping.revision))
    (directory / "map_config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_map(directory: Path) -> VoxelMap:
    raw = json.loads((directory / "map_config.json").read_text())
    config = MapConfig(tuple(raw["origin"]), tuple(raw["shape"]), raw["voxel_m"],
                       raw["sample_stride"], raw["min_depth_m"], raw["max_depth_m"])
    result = VoxelMap(config)
    with np.load(directory / "map.npz") as data:
        for field in ("hits", "frees", "point_sum", "color_sum"):
            value = data[field]
            if value.shape != getattr(result, field).shape:
                raise ValueError("Map array shape does not match metadata")
            setattr(result, field, value.copy())
        result.revision = int(data["revision"])
    return result


def save_tsdf(directory: Path, tsdf: DenseTsdf) -> None:
    np.savez_compressed(directory / "tsdf.npz", value=tsdf.value, weight=tsdf.weight)


def load_tsdf(directory: Path, config: MapConfig) -> DenseTsdf:
    result = DenseTsdf(config)
    with np.load(directory / "tsdf.npz") as data:
        if data["value"].shape != config.shape or data["weight"].shape != config.shape:
            raise ValueError("TSDF grid does not match map configuration")
        result.value[:] = data["value"]
        result.weight[:] = data["weight"]
    return result


def save_mesh(path: Path, mesh: SurfaceMesh) -> None:
    """Binary little-endian PLY with metric world coordinates."""
    vertices = np.asarray(mesh.vertices, dtype="<f4")
    faces = np.asarray(mesh.triangles, dtype="<i4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())
        record = np.empty(len(faces), dtype=[("count", "u1"), ("index", "<i4", (3,))])
        record["count"] = 3
        record["index"] = faces
        handle.write(record.tobytes())
