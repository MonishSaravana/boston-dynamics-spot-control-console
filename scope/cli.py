"""Offline mapping CLI. No Spot SDK import or robot command path."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .data import RgbdFrame
from .evaluate import save_metrics, synthetic_metrics
from .mapping import DenseTsdf, MapConfig, VoxelMap
from .storage import (load_episode, load_map, load_tsdf, save_episode, save_map,
                      save_tsdf)
from .synthetic import SyntheticRoom
from .tum import TumRgbd
from .viewer import open_recording, record_viewer


def _default_output(source: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{source}-{stamp}"


def _run_map(args: argparse.Namespace) -> int:
    if args.source == "synthetic":
        sensor = SyntheticRoom(frames=args.frames)
    elif args.source == "tum":
        if not args.dataset:
            raise SystemExit("scope map tum requires the extracted TUM RGB-D directory")
        sensor = TumRgbd(Path(args.dataset), max_frames=args.frames,
                         frame_stride=args.frame_stride, width=args.width)
    elif args.source == "episode":
        if not args.dataset:
            raise SystemExit("scope map episode requires a saved run directory")
        saved_frames = load_episode(Path(args.dataset))
        sensor = saved_frames[:args.frames]
    else:
        raise SystemExit(f"Unknown source {args.source}")
    frames = list(sensor)
    if not frames:
        raise SystemExit("No synchronized RGB-D frames with supplied poses were found")
    output = args.output or _default_output(args.source)
    output.mkdir(parents=True, exist_ok=True)
    config = MapConfig.synthetic() if args.source == "synthetic" else MapConfig.around_frames(
        frames, voxel_m=args.voxel_m)
    if args.source == "synthetic" and args.voxel_m != config.voxel_m:
        low = np.asarray(config.origin)
        high = low + np.asarray(config.shape) * config.voxel_m
        shape = tuple(int(v) for v in np.ceil((high - low) / args.voxel_m))
        config = replace(config, shape=shape, voxel_m=args.voxel_m)

    source_name = sensor.name if args.source != "episode" else "saved-episode"
    print(f"Mapping {len(frames)} {source_name} frames into {config.shape} voxels", flush=True)
    save_episode(output, frames, source_name)
    tsdf = DenseTsdf(config)
    check_map = VoxelMap(config)
    valid_count = 0
    for frame in frames:
        if not check_map.integrate(frame):
            print(f"Rejected frame: {check_map.rejected[-1]}")
            continue
        tsdf.integrate(frame)
        valid_count += 1
    if valid_count == 0:
        raise SystemExit("No valid RGB-D frames to integrate")
    mesh = tsdf.mesh()
    save_tsdf(output, tsdf)
    if len(mesh.vertices):
        from .storage import save_mesh
        save_mesh(output / "tsdf_mesh.ply", mesh)
    mapping = record_viewer(frames, config, output, mesh)
    if not np.array_equal(mapping.states(), check_map.states()):
        raise RuntimeError("Mapping disagreed between fusion and viewer replay")
    import rerun as rr
    rr.disconnect()
    save_map(output, mapping)
    loaded = load_map(output)
    loaded_tsdf = load_tsdf(output, config)
    replay_frames = load_episode(output)
    if (not np.array_equal(loaded.states(), mapping.states()) or
            not np.array_equal(loaded_tsdf.weight, tsdf.weight) or
            len(replay_frames) != len(frames)):
        raise RuntimeError("Saved map or episode failed round-trip verification")
    metrics: dict[str, int | float | bool | str] = {
        "source": source_name, "frames_total": len(frames),
        "frames_integrated": mapping.revision, "frames_rejected": len(mapping.rejected),
        "alignment_warnings": len(mapping.warnings),
        "tsdf_triangles": len(mesh.triangles),
        "saved_map_roundtrip": True,
    }
    if args.source == "synthetic":
        # A denser independent ray sample is the visibility reference for the
        # downsampled online map; analytic box overlap checks actual geometry.
        reference = VoxelMap(replace(config, sample_stride=1))
        for frame in frames:
            reference.integrate(frame)
        metrics.update(synthetic_metrics(mapping, sensor.boxes, reference))
    save_metrics(output, metrics)
    (output / "diagnostics.json").write_text(json.dumps({
        "rejected": mapping.rejected, "warnings": mapping.warnings}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved run: {output.resolve()}")
    if not args.no_viewer:
        open_recording(output / "reconstruction.rrd")
        print("Opened the Rerun timeline. Press Play to watch the map grow.")
    return 0


def _run_view(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    saved = load_map(directory)
    frames = load_episode(directory)
    if not (directory / "reconstruction.rrd").exists():
        record_viewer(frames, saved.config, directory)
        import rerun as rr
        rr.disconnect()
    print(f"Loaded {saved.revision} map revisions and {len(frames)} frames")
    open_recording(directory / "reconstruction.rrd")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scope", description="Offline known-pose RGB-D mapping")
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map", help="Reconstruct a known-pose RGB-D episode")
    mapping.add_argument("source", choices=("synthetic", "tum", "episode"))
    mapping.add_argument("dataset", nargs="?", help="Extracted TUM RGB-D directory")
    mapping.add_argument("--output", type=Path)
    mapping.add_argument("--frames", type=int, default=28)
    mapping.add_argument("--frame-stride", type=int, default=5,
                         help="For TUM, use every Nth RGB frame")
    mapping.add_argument("--width", type=int, default=320,
                         help="For TUM, downsample RGB/depth to this width")
    mapping.add_argument("--voxel-m", type=float, default=.10)
    mapping.add_argument("--no-viewer", action="store_true")
    viewing = commands.add_parser("view", help="Open a saved mapping run")
    viewing.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "map":
        if args.frames <= 0 or args.voxel_m <= 0 or args.frame_stride <= 0 or args.width <= 0:
            parser.error("frame count, stride, width and voxel size must be positive")
        return _run_map(args)
    return _run_view(args)


if __name__ == "__main__":
    raise SystemExit(main())
