"""Inspectable semantic evidence stored beside, never inside, the map snapshot."""

import json
from pathlib import Path

import numpy as np

from .semantic_pipeline import SemanticRun


def save_semantic_run(directory: Path, run: SemanticRun) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    detection_rows = []
    masks = {}
    for detections in run.detections:
        for detection in detections:
            mask_key = f"mask_{len(detection_rows):05d}"
            masks[mask_key] = detection.mask.astype(np.uint8)
            detection_rows.append({
                "observation_id": detection.observation_id,
                "frame_id": detection.frame_id,
                "source": detection.source,
                "label": detection.label,
                "class_probabilities": detection.class_probabilities,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.box_xyxy,
                "mask_pixels": int(detection.mask.sum()),
                "mask_key": mask_key,
                "truth_id_for_evaluation": detection.truth_id,
            })
    np.savez_compressed(directory / "semantic_masks.npz", **masks)
    (directory / "detections.json").write_text(json.dumps(detection_rows, indent=2) + "\n")
    projections = [{
        "observation_id": item.observation_id,
        "frame_id": item.detection.frame_id,
        "timestamp_s": item.timestamp_s,
        "center_m": item.center.tolist(),
        "bounds_low_m": item.low.tolist(),
        "bounds_high_m": item.high.tolist(),
        "covariance_m2": item.covariance.tolist(),
        "support_pixels": item.support_pixels,
        "valid_depth_fraction": item.valid_depth_fraction,
        "depth_spread_m": item.depth_spread_m,
    } for frame in run.projections for item in frame]
    (directory / "object_observations_3d.json").write_text(
        json.dumps(projections, indent=2) + "\n")
    (directory / "projection_rejections.json").write_text(
        json.dumps(run.rejected_projection_ids, indent=2) + "\n")
    if run.store is not None:
        (directory / "entities.json").write_text(
            json.dumps(run.store.summaries(), indent=2) + "\n")
        (directory / "association.json").write_text(
            json.dumps(run.store.decisions, indent=2) + "\n")


def load_entity_evidence(directory: Path, entity_id: str) -> dict:
    entities = json.loads((directory / "entities.json").read_text())
    entity = next((item for item in entities if item["entity_id"] == entity_id), None)
    if entity is None:
        raise ValueError(f"No entity {entity_id} in {directory}")
    decisions = json.loads((directory / "association.json").read_text())
    detections = {item["observation_id"]: item for item in json.loads(
        (directory / "detections.json").read_text())}
    relation_path = directory / "relations.json"
    relations = json.loads(relation_path.read_text()) if relation_path.exists() else []
    return {"entity": entity,
            "detections": [detections[oid] for oid in entity["observation_ids"]],
            "association_decisions": [item for item in decisions
                                      if item["selected"]["entity_id"] == entity_id],
            "relations": [item for item in relations if item["source"] == entity_id]}
