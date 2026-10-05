"""Traceable object phrases and geometric evidence, without a language planner."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .interaction import TextCommand, TextResponse


class QueryState(StrEnum):
    PENDING = "PENDING"
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STALE = "STALE"
    TRACK_LOST = "TRACK_LOST"
    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


# A small, inspectable spelling/usage aid. The detector's learned text encoder
# handles all other phrases; aliases are not a closed object vocabulary.
ALIASES = (("sofa", "couch"), ("trash can", "garbage bin"),
           ("monitor", "screen"), ("cell phone", "phone"), ("backpack", "bag"))


@dataclass(frozen=True)
class SemanticQuery:
    raw_phrase: str
    normalized: str
    alternatives: tuple[str, ...]
    request_id: str | None = None

    @property
    def entity_label(self):
        for group in ALIASES:
            if any(self.normalized==word or self.normalized.endswith(" "+word) for word in group):
                return group[0]
        return self.normalized

    @classmethod
    def from_command(cls, command: TextCommand):
        raw = command.text
        normalized = " ".join(raw.strip().casefold().split()).rstrip(".!?")
        if not normalized or len(normalized) > 180:
            raise ValueError("Enter an object phrase of 1–180 characters")
        alternatives = [normalized]
        for group in ALIASES:
            for word in group:
                if normalized == word or normalized.endswith(" "+word):
                    prefix = normalized[:-len(word)]
                    alternatives.extend(prefix+other for other in group if other != word)
        return cls(raw, normalized, tuple(dict.fromkeys(alternatives)), command.request_id)


@dataclass
class ObjectCandidate:
    box_xyxy: tuple[float, float, float, float]
    score: float
    detector_match: str
    model: str
    mask: np.ndarray | None = None
    mask_quality: float | None = None
    track_id: str | None = None
    entity_id: str | None = None
    projection: dict | None = None
    evidence: dict = field(default_factory=dict)

    def summary(self):
        return {"box_xyxy": self.box_xyxy, "score": self.score,
                "detector_match": self.detector_match, "model": self.model,
                "mask_available": self.mask is not None, "mask_pixels": int(self.mask.sum()) if self.mask is not None else 0,
                "mask_quality": self.mask_quality, "track_id": self.track_id,
                "entity_id": self.entity_id, "projection": self.projection, "evidence": self.evidence}


@dataclass
class QueryResult:
    query: SemanticQuery
    state: QueryState
    reason: str
    candidates: list[ObjectCandidate]
    timestamp_s: float
    frame_id: str
    metadata: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    def summary(self, now_s=None):
        import time
        consumed=self.metadata.get("result_consumed_monotonic_s")
        return {"user_phrase": self.query.raw_phrase, "normalized_query": self.query.normalized,
                "expanded_queries": self.query.alternatives, "request_id": self.query.request_id,
                "state": str(self.state), "reason": self.reason,
                "source_timestamp_s": self.timestamp_s, "frame_id": self.frame_id,
                "age_s": None if now_s is None else max(0., now_s-self.timestamp_s),
                "result_age_s":None if consumed is None else max(0.,time.monotonic()-consumed),
                "candidates": [c.summary() for c in self.candidates], "model": self.metadata,
                "timings_ms": self.timings_ms,
                "score_meaning": "Uncalibrated detector evidence, not probability of correctness"}

    def response(self):
        return TextResponse(f'{self.state}: {self.query.raw_phrase} — {self.reason}', self.query.request_id)


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(inter/max(1e-9, np.maximum(0., a[2:]-a[:2]).prod()+np.maximum(0., b[2:]-b[:2]).prod()-inter))


def deduplicate(candidates, iou=.65):
    kept = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if not any(box_iou(candidate.box_xyxy, old.box_xyxy) >= iou for old in kept):
            kept.append(candidate)
    return kept


def decide(candidates, threshold=.4, low_threshold=.15):
    candidates = deduplicate(candidates)
    supported = [c for c in candidates if c.score >= threshold]
    if len(supported) > 1:
        return QueryState.AMBIGUOUS, "Multiple supported regions; no object chosen automatically", supported
    if supported:
        return QueryState.FOUND, "One supported region; detector evidence is uncalibrated", supported
    weak = [c for c in candidates if c.score >= low_threshold]
    if weak:
        return QueryState.LOW_CONFIDENCE, "No region passes the acceptance threshold", weak
    return QueryState.NOT_FOUND, "No sufficient visual evidence in this image", []
