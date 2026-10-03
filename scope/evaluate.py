"""Synthetic reconstruction measures against analytic scene geometry."""

import json
from pathlib import Path

import numpy as np

from .mapping import FREE, OCCUPIED, UNKNOWN, MapConfig, VoxelMap
from .synthetic import Box, geometry_occupied


def surface_distance(points: np.ndarray, boxes: tuple[Box, ...]) -> np.ndarray:
    """Euclidean distance to the closest face of any axis-aligned scene box."""
    best = np.full(len(points), np.inf)
    for box in boxes:
        low, high = np.asarray(box.low), np.asarray(box.high)
        outside = np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0),
                                 axis=1)
        inside = np.minimum(points - low, high - points).min(axis=1)
        distance = np.where(outside > 0, outside, inside)
        best = np.minimum(best, distance)
    return best


def occupied_truth(config: MapConfig, boxes: tuple[Box, ...]) -> np.ndarray:
    """Voxel-box overlap, not center-in-box, to account for thin surfaces."""
    coords = np.indices(config.shape).reshape(3, -1).T
    low = np.asarray(config.origin) + coords * config.voxel_m
    high = low + config.voxel_m
    truth = np.zeros(len(coords), dtype=bool)
    for box in boxes:
        truth |= np.all((high > box.low) & (low < box.high), axis=1)
    return truth.reshape(config.shape)


def synthetic_metrics(mapping: VoxelMap, boxes: tuple[Box, ...],
                      reference: VoxelMap | None = None) -> dict[str, float | int]:
    points, _, counts = mapping.cloud()
    if not len(points):
        raise ValueError("No reconstructed points")
    dist = surface_distance(points, boxes)
    state = mapping.states()
    actual_occupied = occupied_truth(mapping.config, boxes)
    pred_occ = state == OCCUPIED
    pred_free = state == FREE
    metrics: dict[str, float | int] = {
        "frames_integrated": mapping.revision,
        "surface_voxels": len(points),
        "surface_error_mean_m": float(dist.mean()),
        "surface_error_p95_m": float(np.percentile(dist, 95)),
        "surface_within_0_15m": float(np.mean(dist <= .15)),
        "occupied_precision": float(np.mean(actual_occupied[pred_occ])) if pred_occ.any() else 0.,
        "free_space_precision": float(np.mean(~actual_occupied[pred_free])) if pred_free.any() else 0.,
        "unknown_fraction": float(np.mean(state == UNKNOWN)),
    }
    if reference is not None:
        ref_state = reference.states()
        ref_occ = ref_state == OCCUPIED
        ref_free = ref_state == FREE
        ref_unknown = ref_state == UNKNOWN
        metrics.update({
            "surface_completeness_vs_dense_rays": float(
                np.sum(pred_occ & ref_occ) / max(1, np.sum(ref_occ))),
            "free_recall_vs_dense_rays": float(
                np.sum(pred_free & ref_free) / max(1, np.sum(ref_free))),
            "unknown_correctness_vs_dense_rays": float(
                np.sum((state == UNKNOWN) & ref_unknown) / max(1, np.sum(state == UNKNOWN))),
            "unknown_recall_vs_dense_rays": float(
                np.sum((state == UNKNOWN) & ref_unknown) / max(1, np.sum(ref_unknown))),
        })
    return metrics


def save_metrics(directory: Path, metrics: dict) -> None:
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if "occupied_precision" not in metrics:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ("Occupied precision", "Free precision", "Unknown correctness",
             "Surface completeness")
    keys = ("occupied_precision", "free_space_precision",
            "unknown_correctness_vs_dense_rays",
            "surface_completeness_vs_dense_rays")
    values = [metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    bars = ax.barh(names, values, color=("#e9a153", "#3299a5", "#697581", "#79bad4"))
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction")
    ax.set_title("Synthetic known-pose reconstruction")
    for bar, value in zip(bars, values):
        label_inside = value > .88
        ax.text(value - .015 if label_inside else value + .012,
                bar.get_y() + bar.get_height()/2, f"{value:.1%}",
                ha="right" if label_inside else "left", va="center", fontsize=9)
    fig.savefig(directory / "quality_report.png", dpi=160)
    plt.close(fig)
