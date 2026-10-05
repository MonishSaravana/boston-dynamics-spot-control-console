"""Episode-local multi-view fusion of semantic 3D observations."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .objects import ObjectObservation3D


def _aabb_iou(a_low: np.ndarray, a_high: np.ndarray,
              b_low: np.ndarray, b_high: np.ndarray) -> float:
    overlap = np.maximum(0, np.minimum(a_high, b_high) - np.maximum(a_low, b_low))
    intersection = float(np.prod(overlap))
    va = float(np.prod(np.maximum(a_high - a_low, .02)))
    vb = float(np.prod(np.maximum(b_high - b_low, .02)))
    return intersection / max(va + vb - intersection, 1e-9)


@dataclass
class WorldEntity:
    entity_id: str
    class_evidence: dict[str, float]
    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    covariance: np.ndarray
    points_world: np.ndarray
    observation_ids: list[str]
    first_seen_s: float
    last_seen_s: float
    supporting_views: set[str]
    depth_valid_fraction: float

    @property
    def class_probabilities(self) -> dict[str, float]:
        total = sum(self.class_evidence.values())
        return {name: value / total for name, value in self.class_evidence.items()}

    @property
    def label(self) -> str:
        return max(self.class_probabilities, key=self.class_probabilities.get)

    @property
    def confidence(self) -> float:
        return float(max(self.class_probabilities.values()) *
                     (1.0 - np.exp(-len(self.supporting_views) / 2.0)) *
                     self.depth_valid_fraction)

    def summary(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "class_probabilities": self.class_probabilities,
            "confidence": self.confidence,
            "center_m": self.center.tolist(),
            "bounds_low_m": self.low.tolist(),
            "bounds_high_m": self.high.tolist(),
            "dimensions_m": (self.high - self.low).tolist(),
            "covariance_m2": self.covariance.tolist(),
            "observation_ids": self.observation_ids,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "supporting_views": len(self.supporting_views),
            "depth_valid_fraction": self.depth_valid_fraction,
        }


@dataclass(frozen=True)
class MatchEvidence:
    observation_id: str
    entity_id: str
    score: float
    class_affinity: float
    center_distance_m: float
    normalized_center_distance: float
    extent_iou: float
    point_overlap: float
    extent_gap_m: float
    accepted: bool
    reason: str

    def summary(self) -> dict:
        return vars(self).copy()


def match_evidence(obs: ObjectObservation3D, entity: WorldEntity, *, point_tree=None) -> MatchEvidence:
    probs = entity.class_probabilities
    class_affinity = sum(value * probs.get(name, 0.)
                         for name, value in obs.detection.class_probabilities.items())
    distance = float(np.linalg.norm(obs.center - entity.center))
    extent_scale = max(.3, float(np.linalg.norm(obs.high - obs.low)),
                       float(np.linalg.norm(entity.high - entity.low)))
    normalized_distance = distance / extent_scale
    gap = float(np.linalg.norm(np.maximum(
        np.maximum(obs.low - entity.high, entity.low - obs.high), 0)))
    overlap = _aabb_iou(obs.low - .03, obs.high + .03,
                        entity.low - .03, entity.high + .03)
    sample = obs.points_world[::max(1, len(obs.points_world) // 400)]
    nearest = (point_tree if point_tree is not None else cKDTree(entity.points_world)).query(sample, workers=1)[0]
    point_overlap = float(np.mean(nearest < .12))
    center_score = np.exp(-.5 * (normalized_distance / .45)**2)
    score = float(.20 * class_affinity + .30 * center_score +
                  .25 * min(1., overlap * 3) + .25 * point_overlap)
    if class_affinity < .12:
        reason = "semantic gate"
    elif normalized_distance > .85:
        reason = "center gate"
    elif gap > .18:
        reason = "extent gap gate"
    elif overlap < .035 and point_overlap < .10:
        reason = "no extent or point support"
    elif score < .43:
        reason = "combined score"
    else:
        reason = "matched"
    return MatchEvidence(obs.observation_id, entity.entity_id, score,
                         float(class_affinity), distance, normalized_distance,
                         overlap, point_overlap, gap, reason == "matched", reason)


class EntityStore:
    """Assign one observation per entity per frame; retain matching evidence."""

    def __init__(self):
        self.entities: dict[str, WorldEntity] = {}
        self.decisions: list[dict] = []
        self.observations: dict[str, ObjectObservation3D] = {}
        self.observation_entity: dict[str, str] = {}
        self._class_counters: dict[str, int] = {}

    def _new(self, obs: ObjectObservation3D) -> WorldEntity:
        label = obs.label.replace(" ", "_")
        self._class_counters[label] = self._class_counters.get(label, 0) + 1
        entity_id = f"{label}_{self._class_counters[label]:02d}"
        weight = max(.05, obs.detection.confidence * obs.valid_depth_fraction)
        entity = WorldEntity(
            entity_id, {k: weight * v for k, v in obs.detection.class_probabilities.items()},
            obs.center.copy(), obs.low.copy(), obs.high.copy(), obs.covariance.copy(),
            obs.points_world.copy(), [obs.observation_id], obs.timestamp_s,
            obs.timestamp_s, {obs.detection.frame_id}, obs.valid_depth_fraction)
        self.entities[entity_id] = entity
        return entity

    @staticmethod
    def _fuse(entity: WorldEntity, obs: ObjectObservation3D) -> None:
        weight = max(.05, obs.detection.confidence * obs.valid_depth_fraction)
        for name, probability in obs.detection.class_probabilities.items():
            entity.class_evidence[name] = entity.class_evidence.get(name, 0.) + weight * probability
        old_center = entity.center.copy()
        n = len(entity.observation_ids)
        entity.low = np.minimum(entity.low, obs.low)
        entity.high = np.maximum(entity.high, obs.high)
        entity.center = ((entity.low + entity.high) / 2.0).astype(np.float32)
        delta = (obs.center - old_center).reshape(3, 1)
        entity.covariance = ((entity.covariance * n + obs.covariance) / (n + 1) +
                             (delta @ delta.T) / (n + 1)**2).astype(np.float32)
        entity.points_world = np.concatenate((entity.points_world, obs.points_world))
        if len(entity.points_world) > 12000:
            chosen = np.linspace(0, len(entity.points_world) - 1, 12000, dtype=int)
            entity.points_world = entity.points_world[chosen]
        entity.observation_ids.append(obs.observation_id)
        entity.last_seen_s = max(entity.last_seen_s, obs.timestamp_s)
        entity.supporting_views.add(obs.detection.frame_id)
        entity.depth_valid_fraction = (entity.depth_valid_fraction * n +
                                       obs.valid_depth_fraction) / (n + 1)

    def add_frame(self, observations: list[ObjectObservation3D]) -> list[WorldEntity]:
        if not observations:
            return []
        if len({o.detection.frame_id for o in observations}) != 1:
            raise ValueError("add_frame expects observations from one frame")
        if len({o.observation_id for o in observations}) != len(observations):
            raise ValueError("Duplicate observation ID")
        prior = list(self.entities.values())
        trees = {entity.entity_id: cKDTree(entity.points_world) for entity in prior}
        candidates = [[match_evidence(obs, entity, point_tree=trees[entity.entity_id]) for entity in prior]
                      for obs in observations]
        assignments: dict[int, int] = {}
        if prior:
            cost = np.full((len(observations), len(prior)), 1e6)
            for i, row in enumerate(candidates):
                for j, evidence in enumerate(row):
                    if evidence.accepted:
                        cost[i, j] = 1. - evidence.score
            rows, cols = linear_sum_assignment(cost)
            assignments = {int(i): int(j) for i, j in zip(rows, cols)
                           if cost[i, j] < 1e5}
        updated = []
        for i, obs in enumerate(observations):
            if obs.observation_id in self.observations:
                raise ValueError(f"Duplicate observation ID: {obs.observation_id}")
            if i in assignments:
                evidence = candidates[i][assignments[i]]
                entity = prior[assignments[i]]
                self._fuse(entity, obs)
                selected = evidence.summary()
            else:
                entity = self._new(obs)
                selected = {"observation_id": obs.observation_id,
                            "entity_id": entity.entity_id, "reason": "new entity",
                            "accepted": True, "score": None}
            self.observations[obs.observation_id] = obs
            self.observation_entity[obs.observation_id] = entity.entity_id
            self.decisions.append({"selected": selected,
                                   "candidates": [item.summary() for item in candidates[i]]})
            updated.append(entity)
        return updated

    def summaries(self) -> list[dict]:
        return [entity.summary() for entity in self.entities.values()]
