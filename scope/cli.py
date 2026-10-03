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


def _semantic_detector(args: argparse.Namespace, sensor, frames: list[RgbdFrame]):
    if args.detector == "truth" or (args.detector is None and frames[0].source == "SIMULATED"):
        if frames[0].source != "SIMULATED":
            raise SystemExit("Synthetic truth masks require synthetic frames")
        from .truth import SyntheticTruthDetector
        from .synthetic import room_boxes
        return SyntheticTruthDetector(sensor.boxes if hasattr(sensor, "boxes") else room_boxes())
    from .detector import TorchvisionMaskDetector
    classes = {name.strip() for name in args.classes.split(",") if name.strip()} \
        if args.classes else None
    return TorchvisionMaskDetector(score_threshold=args.score_threshold,
                                   include_classes=classes)


def _semantic_run(args: argparse.Namespace, sensor, frames: list[RgbdFrame]):
    from .semantic_pipeline import run_semantics
    detector = _semantic_detector(args, sensor, frames)
    return run_semantics(frames, detector, stage="entities" if args.command == "map"
                         else {"detect": "detect", "project-objects": "project",
                               "entities": "entities"}[args.command])


def _save_semantics(output: Path, run, sensor) -> None:
    from .semantic_storage import save_semantic_run
    save_semantic_run(output, run)
    if run.frames[0].source == "SIMULATED":
        from .object_eval import evaluate_synthetic, save_object_metrics
        room = sensor if isinstance(sensor, SyntheticRoom) else SyntheticRoom()
        save_object_metrics(output, evaluate_synthetic(run, room))
    if run.store is not None:
        from .relations import entity_relations
        (output / "relations.json").write_text(json.dumps(
            entity_relations(list(run.store.entities.values())), indent=2) + "\n")


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
    semantic_run = _semantic_run(args, sensor, frames) if args.entities else None
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
    semantic_recorder = None
    if semantic_run is not None:
        from .semantic_viewer import SemanticRecorder
        semantic_recorder = SemanticRecorder(semantic_run)
    semantic_eye = None
    if semantic_run is not None:
        centers = np.array([entity.center for entity in semantic_run.store.entities.values()])
        if len(centers):
            target = (centers.min(axis=0) + centers.max(axis=0)) / 2
            span = max(1.0, float(np.max(np.ptp(centers, axis=0))))
            semantic_eye = (target, span)
    mapping = record_viewer(
        frames, config, output, mesh,
        frame_hook=semantic_recorder.log_frame if semantic_recorder else None,
        semantic_eye=semantic_eye)
    if semantic_run is not None:
        if semantic_recorder.store.summaries() != semantic_run.store.summaries():
            raise RuntimeError("Semantic viewer replay disagreed with entity fusion")
        _save_semantics(output, semantic_run, sensor)
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


def _run_semantic_stage(args: argparse.Namespace) -> int:
    if args.source == "synthetic":
        sensor = SyntheticRoom(frames=args.frames)
    elif args.source == "tum":
        if not args.dataset:
            raise SystemExit(f"scope {args.command} tum requires the extracted TUM directory")
        sensor = TumRgbd(Path(args.dataset), max_frames=args.frames,
                         frame_stride=args.frame_stride, width=args.width)
    else:
        if not args.dataset:
            raise SystemExit(f"scope {args.command} episode requires a saved run directory")
        sensor = load_episode(Path(args.dataset))[:args.frames]
    frames = list(sensor)
    if not frames:
        raise SystemExit("No synchronized RGB-D frames with supplied poses were found")
    output = args.output or _default_output(f"{args.command}-{args.source}")
    output.mkdir(parents=True, exist_ok=True)
    save_episode(output, frames, getattr(sensor, "name", "saved-episode"))
    run = _semantic_run(args, sensor, frames)
    _save_semantics(output, run, sensor)
    from .semantic_viewer import record_semantic_only
    replay_store = record_semantic_only(run, output)
    if run.store is not None and replay_store.summaries() != run.store.summaries():
        raise RuntimeError("Semantic viewer replay disagreed with entity fusion")
    print(f"{run.stage}: {sum(map(len, run.detections))} masks, "
          f"{sum(map(len, run.projections))} projected, "
          f"{len(run.store.entities) if run.store else 0} entities")
    if run.store is not None:
        for entity in run.store.entities.values():
            print(f"{entity.entity_id} [{entity.confidence:.0%}] "
                  f"center={entity.center.round(2).tolist()} m "
                  f"views={len(entity.supporting_views)}")
    print(f"Saved run: {output.resolve()}")
    if not args.no_viewer:
        open_recording(output / "semantic.rrd")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    from .semantic_storage import load_entity_evidence
    print(json.dumps(load_entity_evidence(Path(args.directory), args.entity_id), indent=2))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    from .object_eval import robustness_benchmark, save_robustness
    output = args.output or _default_output("semantic-benchmark")
    cases = robustness_benchmark()
    save_robustness(output, cases)
    print(json.dumps({name: {
        "association_pair_accuracy": metrics["association_pair_accuracy"],
        "false_merges": metrics["false_merges"],
        "false_splits": metrics["false_splits"],
        "entity_localization_error_mean_m": metrics["entity_localization_error_mean_m"],
        "projection_rejections": metrics["projection_rejections"],
        "projected_observations": metrics["projected_observations"],
    } for name, metrics in cases.items()}, indent=2))
    print(f"Saved benchmark: {output.resolve()}")
    return 0


def _run_view(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if (directory / "semantic.rrd").exists() and not (directory / "map.npz").exists():
        frames = load_episode(directory)
        print(f"Loaded {len(frames)} semantic frames")
        open_recording(directory / "semantic.rrd")
        return 0
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
    mapping.add_argument("--entities", action="store_true",
                         help="Overlay episode-local semantic entities on the map")
    mapping.add_argument("--detector", choices=("truth", "torchvision"))
    mapping.add_argument("--score-threshold", type=float, default=.55)
    mapping.add_argument("--classes", help="Comma-separated model classes to retain")
    for name in ("detect", "project-objects", "entities"):
        stage = commands.add_parser(name, help=f"Inspect the {name} semantic stage")
        stage.add_argument("source", choices=("synthetic", "tum", "episode"))
        stage.add_argument("dataset", nargs="?")
        stage.add_argument("--output", type=Path)
        stage.add_argument("--frames", type=int, default=10)
        stage.add_argument("--frame-stride", type=int, default=10)
        stage.add_argument("--width", type=int, default=320)
        stage.add_argument("--detector", choices=("truth", "torchvision"))
        stage.add_argument("--score-threshold", type=float, default=.55)
        stage.add_argument("--classes", help="Comma-separated model classes to retain")
        stage.add_argument("--no-viewer", action="store_true")
    viewing = commands.add_parser("view", help="Open a saved mapping run")
    viewing.add_argument("directory")
    inspecting = commands.add_parser("inspect", help="Show one entity and its supporting evidence")
    inspecting.add_argument("directory")
    inspecting.add_argument("entity_id")
    benchmark = commands.add_parser("benchmark-entities",
                                    help="Run controlled synthetic semantic failures")
    benchmark.add_argument("--output", type=Path)
    from .memory_cli import register_memory_commands, run_memory_command
    register_memory_commands(commands)
    args = parser.parse_args(argv)
    if args.command == "map":
        if args.frames <= 0 or args.voxel_m <= 0 or args.frame_stride <= 0 or args.width <= 0:
            parser.error("frame count, stride, width and voxel size must be positive")
        return _run_map(args)
    if args.command in ("detect", "project-objects", "entities"):
        if args.frames <= 0 or args.frame_stride <= 0 or args.width <= 0:
            parser.error("frame count, stride, and width must be positive")
        return _run_semantic_stage(args)
    if args.command == "inspect":
        return _run_inspect(args)
    if args.command == "benchmark-entities":
        return _run_benchmark(args)
    if args.command in ("memory", "memory-demo", "compare-episodes", "link-entities", "history",
                        "last-seen", "replay-memory", "benchmark-memory"):
        return run_memory_command(args)
    return _run_view(args)


if __name__ == "__main__":
    raise SystemExit(main())
