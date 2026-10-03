"""Synthetic semantic-object metrics and repeatable robustness probes."""

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .entities import EntityStore
from .evaluate import surface_distance
from .objects import project_object
from .semantic_pipeline import SemanticRun, run_semantics
from .synthetic import Box, SyntheticRoom
from .truth import SyntheticTruthDetector, semantic_identity


def _truth_parts(boxes: tuple[Box, ...], identity: str) -> tuple[Box, ...]:
    return tuple(box for box in boxes if (tag := semantic_identity(box.name)) is not None
                 and tag[0] == identity)


def _truth_bounds(boxes: tuple[Box, ...], identity: str) -> tuple[np.ndarray, np.ndarray]:
    parts = _truth_parts(boxes, identity)
    return (np.min([part.low for part in parts], axis=0),
            np.max([part.high for part in parts], axis=0))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a & b) / max(1, np.sum(a | b)))


def evaluate_synthetic(run: SemanticRun, room: SyntheticRoom,
                       truth_frames: list | None = None) -> dict[str, float | int | None]:
    truth = SyntheticTruthDetector(room.boxes)
    truth_frames = truth_frames or run.frames
    detector_tp = detector_fp = detector_fn = 0
    mask_ious = []
    for frame, predictions in zip(truth_frames, run.detections):
        actual = truth.detect(frame)
        if not predictions or not actual:
            detector_fp += len(predictions)
            detector_fn += len(actual)
            continue
        quality = np.zeros((len(predictions), len(actual)))
        for i, detected in enumerate(predictions):
            for j, expected in enumerate(actual):
                if detected.label == expected.label:
                    quality[i, j] = _mask_iou(detected.mask, expected.mask)
        rows, cols = linear_sum_assignment(1 - quality)
        matched = [(int(i), int(j)) for i, j in zip(rows, cols)
                   if quality[i, j] >= .5]
        detector_tp += len(matched)
        detector_fp += len(predictions) - len(matched)
        detector_fn += len(actual) - len(matched)
        mask_ious.extend(quality[i, j] for i, j in matched)
    metrics: dict[str, float | int | None] = {
        "detector_precision_at_iou_0_5": detector_tp / max(1, detector_tp + detector_fp),
        "detector_recall_at_iou_0_5": detector_tp / max(1, detector_tp + detector_fn),
        "mask_iou_mean_matched": float(np.mean(mask_ious)) if mask_ious else 0.,
        "detections": sum(map(len, run.detections)),
    }
    if run.stage == "detect":
        return metrics
    centroid_errors, extent_errors, point_distances = [], [], []
    for projected in (item for frame in run.projections for item in frame):
        identity = projected.detection.truth_id
        if identity is None:
            continue
        low, high = _truth_bounds(room.boxes, identity)
        centroid_errors.append(float(np.linalg.norm(projected.center - (low + high) / 2)))
        extent_errors.append(float(np.mean(np.abs((projected.high - projected.low) -
                                                  (high - low)))))
        point_distances.extend(surface_distance(
            projected.points_world, _truth_parts(room.boxes, identity)))
    metrics.update({
        "projected_observations": sum(map(len, run.projections)),
        "projection_rejections": len(run.rejected_projection_ids),
        "observation_centroid_error_mean_m": float(np.mean(centroid_errors)) if centroid_errors else 0.,
        "observation_extent_error_mean_m": float(np.mean(extent_errors)) if extent_errors else 0.,
        "object_point_precision_within_0_05m": float(np.mean(np.asarray(point_distances) < .05))
        if point_distances else 0.,
    })
    # Independent dense reference is the visible truth-mask point set from the
    # same frames. This is visible-surface coverage, not hidden-shape recovery.
    truth_points = {}
    for frame in truth_frames:
        for detection in truth.detect(frame):
            projected = project_object(frame, detection, max_points=8000)
            if projected is not None:
                truth_points.setdefault(detection.truth_id, []).append(projected.points_world)
    reference = {name: np.concatenate(clouds) for name, clouds in truth_points.items()}
    if run.store is None:
        return metrics
    store = run.store
    truth_to_entities: dict[str, set[str]] = {}
    entity_to_truth: dict[str, set[str]] = {}
    sequences: dict[str, list[tuple[float, str]]] = {}
    for oid, eid in store.observation_entity.items():
        observation = store.observations[oid]
        identity = observation.detection.truth_id
        if identity is None:
            continue
        truth_to_entities.setdefault(identity, set()).add(eid)
        entity_to_truth.setdefault(eid, set()).add(identity)
        sequences.setdefault(identity, []).append((observation.timestamp_s, eid))
    false_merges = sum(len(ids) > 1 for ids in entity_to_truth.values())
    false_splits = sum(max(0, len(ids) - 1) for ids in truth_to_entities.values())
    id_switches = sum(sum(a[1] != b[1] for a, b in zip(sorted(items), sorted(items)[1:]))
                      for items in sequences.values())
    pairs = list(store.observation_entity.items())
    pair_correct = pair_total = 0
    for i, (oid_a, eid_a) in enumerate(pairs):
        truth_a = store.observations[oid_a].detection.truth_id
        if truth_a is None:
            continue
        for oid_b, eid_b in pairs[i+1:]:
            truth_b = store.observations[oid_b].detection.truth_id
            if truth_b is None:
                continue
            pair_total += 1
            pair_correct += (truth_a == truth_b) == (eid_a == eid_b)
    class_correct, localization, class_brier, completeness = [], [], [], []
    for entity in store.entities.values():
        associated = [store.observations[oid].detection.truth_id
                      for oid in entity.observation_ids]
        associated = [identity for identity in associated if identity]
        if not associated:
            continue
        identity = max(set(associated), key=associated.count)
        truth_label = semantic_identity(_truth_parts(room.boxes, identity)[0].name)[1]
        class_correct.append(entity.label == truth_label)
        class_brier.append((1 - entity.class_probabilities.get(truth_label, 0.))**2)
        low, high = _truth_bounds(room.boxes, identity)
        localization.append(float(np.linalg.norm(entity.center - (low + high) / 2)))
        if identity in reference:
            subset = reference[identity][::max(1, len(reference[identity]) // 5000)]
            nearest = cKDTree(entity.points_world).query(subset, workers=1)[0]
            completeness.append(float(np.mean(nearest < .10)))
    metrics.update({
        "entities": len(store.entities),
        "association_pair_accuracy": pair_correct / pair_total if pair_total else None,
        "false_merges": false_merges,
        "false_splits": false_splits,
        "id_switches": id_switches,
        "entity_class_accuracy": float(np.mean(class_correct)) if class_correct else 0.,
        "entity_class_brier_mean": float(np.mean(class_brier)) if class_brier else 0.,
        "entity_localization_error_mean_m": float(np.mean(localization)) if localization else 0.,
        "visible_object_point_completeness_within_0_10m": float(np.mean(completeness))
        if completeness else 0.,
    })
    return metrics


def save_object_metrics(directory: Path, metrics: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "semantic_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if "association_pair_accuracy" not in metrics:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Mask IoU", "Object point precision", "Visible point completeness",
             "Association accuracy", "Class accuracy"]
    keys = ["mask_iou_mean_matched", "object_point_precision_within_0_05m",
            "visible_object_point_completeness_within_0_10m",
            "association_pair_accuracy", "entity_class_accuracy"]
    values = [metrics[key] or 0 for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].barh(names, values, color="#4aa8b5")
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Fraction")
    axes[1].bar(["Observation center", "Entity center", "Extent"],
                [metrics["observation_centroid_error_mean_m"],
                 metrics["entity_localization_error_mean_m"],
                 metrics["observation_extent_error_mean_m"]], color="#e6a056")
    axes[1].set_ylabel("Mean error (m)")
    fig.suptitle("Synthetic semantic-object reconstruction")
    fig.savefig(directory / "semantic_quality.png", dpi=160)
    plt.close(fig)


class _CachedTruth:
    name = "cached-synthetic-truth"

    def __init__(self, detections: dict):
        self.detections = detections

    def detect(self, frame):
        return self.detections[frame.frame_id]


def robustness_benchmark() -> dict[str, dict]:
    """Reproducible depth, occlusion, pose, and same-class stress cases."""
    room = SyntheticRoom(frames=10, width=128, height=96)
    frames = list(room)
    detector = SyntheticTruthDetector(room.boxes)
    masks = {frame.frame_id: detector.detect(frame) for frame in frames}
    rng = np.random.default_rng(20261003)
    cases = {}

    def measure(name, case_frames, case_masks, case_room=room):
        run = run_semantics(case_frames, _CachedTruth(case_masks))
        metrics = evaluate_synthetic(run, case_room,
                                     truth_frames=frames if case_room is room else None)
        metrics["mean_valid_depth_fraction"] = float(np.mean([
            obs.valid_depth_fraction for group in run.projections for obs in group])) \
            if any(run.projections) else 0.
        cases[name] = metrics

    measure("baseline", frames, masks)

    noisy = []
    for frame in frames:
        depth = frame.depth_m.copy()
        valid = depth > 0
        depth[valid] += rng.normal(0, .04, valid.sum()).astype(np.float32)
        depth[rng.random(depth.shape) < .10] = 0
        noisy.append(replace(frame, depth_m=depth))
    measure("depth_noise_4cm_10pct_missing", noisy, masks)

    missing = []
    for frame in frames:
        depth = frame.depth_m.copy()
        depth[rng.random(depth.shape) < .80] = 0
        missing.append(replace(frame, depth_m=depth))
    measure("depth_80pct_missing", missing, masks)

    occluded_masks = {name: list(group) for name, group in masks.items()}
    for frame in frames[2:7]:
        group = []
        for detection in masks[frame.frame_id]:
            if detection.truth_id == "chair_a":
                mask = detection.mask.copy()
                x1, _, x2, _ = detection.box_xyxy
                mask[:, (x1 + x2)//2:] = False
                detection = replace(detection, mask=mask)
            group.append(detection)
        occluded_masks[frame.frame_id] = group
    measure("partial_chair_occlusion", frames, occluded_masks)

    perturbed = list(frames)
    pose = frames[5].T_world_camera.copy()
    pose[:3, 3] += [1.0, 0, 0]
    perturbed[5] = replace(frames[5], T_world_camera=pose)
    measure("pose_shift_1m_one_frame", perturbed, masks)

    disagreement = {name: list(group) for name, group in masks.items()}
    changed = []
    for detection in masks[frames[3].frame_id]:
        if detection.truth_id == "chair_a":
            detection = replace(detection,
                                class_probabilities={"chair": .35, "stool": .65})
        changed.append(detection)
    disagreement[frames[3].frame_id] = changed
    measure("one_view_class_disagreement", frames, disagreement)

    close_room = SyntheticRoom(frames=8, width=128, height=96)
    close_boxes = []
    for part in close_room.boxes:
        if part.name.startswith("table") or part.name == "backpack":
            continue
        if part.name.startswith("chair B"):
            shift = np.array([-1.22, 0., 0.])
            part = Box(part.name, tuple(np.asarray(part.low) + shift),
                       tuple(np.asarray(part.high) + shift), part.color)
        close_boxes.append(part)
    close_room.boxes = tuple(close_boxes)
    close_frames = list(close_room)
    close_detector = SyntheticTruthDetector(close_room.boxes)
    close_masks = {frame.frame_id: close_detector.detect(frame) for frame in close_frames}
    measure("two_chairs_0p2m_gap", close_frames, close_masks, close_room)
    return cases


def save_robustness(directory: Path, cases: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "semantic_robustness.json").write_text(json.dumps(cases, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cases)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].barh(names, [cases[name]["association_pair_accuracy"] or 0 for name in names],
                 color="#4aa8b5")
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Association pair accuracy")
    for index, name in enumerate(names):
        if cases[name]["association_pair_accuracy"] is None:
            axes[0].text(.02, index, "N/A: too few projected observations",
                         va="center", fontsize=8)
    axes[1].barh(names, [cases[name]["entity_localization_error_mean_m"]
                         for name in names], color="#e6a056")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean entity center error (m)")
    fig.suptitle("Controlled synthetic object failures")
    fig.savefig(directory / "semantic_robustness.png", dpi=160)
    plt.close(fig)
