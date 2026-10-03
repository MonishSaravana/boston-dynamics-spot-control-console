"""Run detection, projection, and fusion as independently selectable stages."""

from dataclasses import dataclass

from .data import RgbdFrame
from .entities import EntityStore
from .objects import (ObjectDetector, ObjectObservation2D,
                      ObjectObservation3D, project_object)


@dataclass
class SemanticRun:
    frames: list[RgbdFrame]
    detections: list[list[ObjectObservation2D]]
    projections: list[list[ObjectObservation3D]]
    store: EntityStore | None
    rejected_projection_ids: list[str]
    detector_name: str
    stage: str


def run_semantics(frames: list[RgbdFrame], detector: ObjectDetector,
                  stage: str = "entities") -> SemanticRun:
    if stage not in ("detect", "project", "entities"):
        raise ValueError("stage must be detect, project, or entities")
    detections = []
    projections = []
    rejected = []
    store = EntityStore() if stage == "entities" else None
    for frame in frames:
        found = detector.detect(frame)
        for item in found:
            item.validate(frame)
        detections.append(found)
        projected = []
        if stage != "detect":
            for item in found:
                observation = project_object(frame, item)
                if observation is None:
                    rejected.append(item.observation_id)
                else:
                    projected.append(observation)
        projections.append(projected)
        if store is not None:
            store.add_frame(projected)
    return SemanticRun(frames, detections, projections, store, rejected,
                       detector.name, stage)
