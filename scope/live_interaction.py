"""Explicit, fail-closed target confirmation and destination authorization.

Host monotonic time controls freshness. Source timestamps remain evidence only.
"""
from dataclasses import dataclass, field
from dataclasses import replace
from enum import StrEnum
import math
import logging
import time

import numpy as np

from .mapping import FREE
from .runtime import distribution


class SelectionSource(StrEnum):
    QUERY = "QUERY"
    POINTING = "POINTING"
    DIRECT = "DIRECT"


class InteractionError(ValueError):
    pass


@dataclass(frozen=True)
class TargetCandidate:
    entity_id: str
    label: str
    source: SelectionSource
    score: float  # Uncalibrated evidence, not probability of correctness.
    observed_host_s: float
    source_timestamp_s: float | None
    bounds_low_m: tuple[float, float, float] | None
    bounds_high_m: tuple[float, float, float] | None
    world_frame: str | None
    camera: str | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    evidence: dict = field(default_factory=dict)
    track_state: str = "FOUND"


@dataclass(frozen=True)
class ConfirmedTarget:
    candidate: TargetCandidate
    confirmed_host_s: float
    revision: int
    anchor_low_m: tuple[float, float, float] | None
    anchor_high_m: tuple[float, float, float] | None


@dataclass(frozen=True)
class RobotPose:
    x_m: float
    y_m: float
    yaw_rad: float
    observed_host_s: float
    world_frame: str
    standing: bool = True


@dataclass(frozen=True)
class DestinationProposal:
    target: ConfirmedTarget
    x_m: float
    y_m: float
    yaw_rad: float
    world_frame: str
    stand_off_m: float
    clearance_m: float
    route_xy_m: tuple[tuple[float, float], ...]
    map_revision: int
    proposed_host_s: float


class VirtualExecutor:
    mode = "SIMULATED"

    def __init__(self):
        self.pose = None
        self.commands = []

    def execute(self, destination):
        self.pose = (destination.x_m, destination.y_m, destination.yaw_rad)
        self.commands.append(destination)
        return "SIMULATED_ARRIVAL"

    def stop(self):
        pass


class DryRunExecutor:
    mode = "DRY_RUN"

    def __init__(self):
        self.proposals = []

    def execute(self, destination):
        self.proposals.append(destination)
        logging.getLogger(__name__).warning(
            "WOULD_EXECUTE_NO_MOTION target=%s world=%s x_m=%.3f y_m=%.3f yaw_rad=%.3f",
            destination.target.candidate.entity_id, destination.world_frame,
            destination.x_m, destination.y_m, destination.yaw_rad)
        return "WOULD_EXECUTE_NO_MOTION"

    def stop(self):
        pass


class TimingBook:
    def __init__(self):
        self.samples = {}

    def add(self, name, duration_ms):
        if math.isfinite(duration_ms) and duration_ms >= 0:
            values = self.samples.setdefault(name, [])
            values.append(float(duration_ms))
            del values[:-1000]

    def snapshot(self):
        return {name: {"current_ms": values[-1], **distribution(values)}
                for name, values in self.samples.items() if values}


def _clear_disk(mapping, x, y, radius):
    """Every footprint/clearance sample must be observed free, including route."""
    cfg = mapping.config
    states = mapping.topdown(min_z=.25, max_z=1.55)
    if not states.size:
        return False
    step = min(cfg.voxel_m / 2, .05)
    offsets = np.arange(-radius, radius + step / 2, step)
    for dx in offsets:
        for dy in offsets:
            if dx * dx + dy * dy > radius * radius:
                continue
            ij = np.floor((np.array([x + dx, y + dy]) -
                           np.asarray(cfg.origin[:2])) / cfg.voxel_m).astype(int)
            i, j = ij
            if (i < 0 or j < 0 or i >= states.shape[0] or
                    j >= states.shape[1] or states[i, j] != FREE):
                return False
    return True


def propose_destination(target, robot, mapping, now_s, stand_off_m=1.2,
                        footprint_m=.42, clearance_m=.15):
    c = target.candidate
    if c.bounds_low_m is None or c.bounds_high_m is None:
        raise InteractionError("DEPTH UNAVAILABLE: no target 3D bounds")
    low, high = np.asarray(c.bounds_low_m), np.asarray(c.bounds_high_m)
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or not np.all(high > low)):
        raise InteractionError("DEPTH UNAVAILABLE: invalid target geometry")
    if not c.world_frame or c.world_frame != robot.world_frame:
        raise InteractionError("POSE UNAVAILABLE: target and robot frames differ")
    if mapping is None or mapping.revision < 1:
        raise InteractionError("MAP UNAVAILABLE")
    if getattr(mapping, "world_frame", None) != c.world_frame:
        raise InteractionError("MAP FRAME UNAVAILABLE or inconsistent")
    if (not robot.standing or not 0 <= now_s - robot.observed_host_s <= .6):
        raise InteractionError("ROBOT STATE STALE or not standing")
    center = (low + high) / 2
    toward_robot = np.array([robot.x_m, robot.y_m]) - center[:2]
    if np.linalg.norm(toward_robot) < .05:
        raise InteractionError("ROBOT POSITION UNRESOLVED")
    toward_robot /= np.linalg.norm(toward_robot)
    for angle in (0., math.pi/6, -math.pi/6, math.pi/3, -math.pi/3):
        direction = np.array([math.cos(angle)*toward_robot[0]-math.sin(angle)*toward_robot[1],
                              math.sin(angle)*toward_robot[0]+math.cos(angle)*toward_robot[1]])
        face = center[:2] + direction * np.linalg.norm((high-low)[:2]) / 2
        goal = face + direction * stand_off_m
        distance = np.linalg.norm(goal - np.array([robot.x_m, robot.y_m]))
        route = tuple(tuple(float(v) for v in point) for point in
                      np.linspace([robot.x_m, robot.y_m], goal,
                                  max(2, int(math.ceil(distance/.1))+1)))
        if all(_clear_disk(mapping, x, y, footprint_m+clearance_m)
               for x, y in route):
            yaw = math.atan2(center[1]-goal[1], center[0]-goal[0])
            return DestinationProposal(target, float(goal[0]), float(goal[1]),
                yaw, c.world_frame, stand_off_m, clearance_m, route,
                mapping.revision, now_s)
    raise InteractionError("DESTINATION BLOCKED or unknown")


class InteractionSession:
    """No detector result dispatches a command; GO is separate and single-use."""

    def __init__(self, executor, clock=time.monotonic):
        self.executor, self.clock = executor, clock
        self.candidates = []
        self.selected = self.confirmed = self.proposal = None
        self.revision = 0
        self.message = "Select a target"
        self.timings = TimingBook()

    def offer(self, candidates, *, ambiguous=False):
        self.candidates = sorted(candidates, key=lambda c: -c.score)
        self.revision += 1
        self.confirmed = self.proposal = None
        self.selected = None if ambiguous else next(iter(self.candidates), None)
        self.message = ("NOT_FOUND" if not self.candidates else
                        "AMBIGUOUS: select a candidate" if ambiguous else
                        "Candidate highlighted; confirm or reject")
        return self.selected

    def select(self, entity_id):
        self.selected = next((c for c in self.candidates if c.entity_id == entity_id), None)
        self.confirmed = self.proposal = None
        if self.selected is None:
            raise InteractionError("Target not in current candidates")
        self.message = "Candidate highlighted; confirm or reject"
        return self.selected

    def refresh_observation(self, candidate):
        """Retain approvals only while a newly observed target has stable geometry."""
        old = self.selected
        if old is None or old.entity_id != candidate.entity_id:
            return False
        stable = (old.bounds_low_m is not None and candidate.bounds_low_m is not None and
                  old.bounds_high_m is not None and candidate.bounds_high_m is not None and
                  old.world_frame == candidate.world_frame and
                  np.linalg.norm(np.asarray(old.bounds_low_m)-candidate.bounds_low_m) <= .1 and
                  np.linalg.norm(np.asarray(old.bounds_high_m)-candidate.bounds_high_m) <= .1 and
                  candidate.track_state in ("FOUND", "VISIBLE", "AMBIGUOUS"))
        if stable and self.confirmed:
            stable = (np.linalg.norm(np.asarray(self.confirmed.anchor_low_m)-candidate.bounds_low_m) <= .1 and
                      np.linalg.norm(np.asarray(self.confirmed.anchor_high_m)-candidate.bounds_high_m) <= .1)
        self.selected = candidate
        self.candidates = [candidate if c.entity_id == candidate.entity_id else c
                           for c in self.candidates]
        if not stable:
            self.confirmed = self.proposal = None
            self.message = "TARGET MOVED or lost; confirm again"
            return False
        if self.confirmed:
            self.confirmed = replace(self.confirmed, candidate=candidate)
            if self.proposal:
                self.proposal = replace(self.proposal, target=self.confirmed)
        return True

    def invalidate(self, reason):
        self.confirmed = self.proposal = None
        self.message = reason

    def reject(self):
        self.candidates = [c for c in self.candidates if c != self.selected]
        self.revision += 1
        self.confirmed = self.proposal = None
        self.selected = next(iter(self.candidates), None)
        self.message = "Next candidate" if self.selected else "NOT_FOUND"
        return self.selected

    def confirm(self):
        c, now = self.selected, self.clock()
        if c is None:
            raise InteractionError("No target selected")
        if c.track_state not in ("FOUND", "VISIBLE", "AMBIGUOUS") or not 0 <= now-c.observed_host_s <= 1.5:
            raise InteractionError("TARGET STALE or lost")
        self.confirmed = ConfirmedTarget(c, now, self.revision,
                                         c.bounds_low_m, c.bounds_high_m)
        self.proposal = None
        self.message = "Target confirmed; preview a destination"
        return self.confirmed

    def propose(self, robot, mapping):
        if self.confirmed is None:
            raise InteractionError("CONFIRM TARGET first")
        now, start = self.clock(), time.perf_counter()
        if not 0 <= now-self.confirmed.candidate.observed_host_s <= 1.5:
            self.confirmed = self.proposal = None
            raise InteractionError("TARGET STALE")
        self.proposal = propose_destination(self.confirmed, robot, mapping, now)
        self.timings.add("destination_generation", (time.perf_counter()-start)*1000)
        self.message = "Destination previewed; GO is separate"
        return self.proposal

    def go(self, robot, mapping, health_check=None):
        proposal, now = self.proposal, self.clock()
        if proposal is None or self.confirmed is None:
            raise InteractionError("No confirmed destination")
        if (self.confirmed.revision != self.revision or
                not 0 <= now-self.confirmed.candidate.observed_host_s <= 1.5 or
                not 0 <= now-proposal.proposed_host_s <= 10.0):
            self.proposal = None
            raise InteractionError("TARGET STALE: GO cancelled")
        if health_check is not None:
            health_check(self.confirmed.candidate)
        fresh = propose_destination(self.confirmed, robot, mapping, now,
                                    proposal.stand_off_m, clearance_m=proposal.clearance_m)
        if math.hypot(fresh.x_m-proposal.x_m, fresh.y_m-proposal.y_m) > .05:
            self.proposal = None
            raise InteractionError("DESTINATION CHANGED: preview again")
        self.proposal = None  # single-use, also when dispatch fails
        start = time.perf_counter()
        result = self.executor.execute(fresh)
        self.timings.add("command_dispatch", (time.perf_counter()-start)*1000)
        self.message = result
        return result

    def stop(self):
        self.proposal = None
        self.executor.stop()
        self.message = "Stop requested; class E-stop remains separate"


def candidates_from_query(query, entities, host_s, world_frame):
    """Bridge existing query output; a 2D-only result cannot get a destination."""
    if query.get("state") not in ("FOUND", "AMBIGUOUS"):
        return []
    by_id = {e["entity_id"]: e for e in entities}
    result = []
    for index, item in enumerate(query.get("candidates", [])):
        entity = by_id.get(item.get("entity_id"))
        result.append(TargetCandidate(item.get("entity_id") or item.get("track_id") or
            f"region:{query.get('frame_id', 'unknown')}:{index}",
            item.get("detector_match") or query.get("normalized_query", "object"),
            SelectionSource.QUERY, float(item["score"]), host_s,
            query.get("source_timestamp_s"),
            tuple(entity["bounds_low_m"]) if entity else None,
            tuple(entity["bounds_high_m"]) if entity else None,
            world_frame if entity else None,
            item.get("evidence", {}).get("camera"),
            tuple(item["box_xyxy"]) if item.get("box_xyxy") else None,
            {"query": query.get("user_phrase"), "detector": item.get("evidence")},
            query.get("state", "STALE")))
    return result


def candidates_from_pointing(ranking, entities, host_s, world_frame, source_s=None):
    if ranking.get("state") == "ABSTAIN":
        return [], True
    by_id = {e["entity_id"]: e for e in entities}
    scores = ranking.get("scores", {})
    known = sorted(((key, value) for key, value in scores.items() if key in by_id),
                   key=lambda row: -row[1])
    unresolved = (not known or known[0][1] < .6 or scores.get("unknown", 0) > .3 or
                  len(known) > 1 and known[0][1]-known[1][1] < .2)
    candidates = [TargetCandidate(key, by_id[key]["label"], SelectionSource.POINTING,
        float(score), host_s, source_s, tuple(by_id[key]["bounds_low_m"]),
        tuple(by_id[key]["bounds_high_m"]), world_frame,
        evidence={"pointing": ranking}) for key, score in known]
    return candidates, unresolved


def direct_candidate(entity, host_s, world_frame, current_source_s=None):
    last_seen = entity.get("last_seen_s")
    fresh = (current_source_s is None or last_seen is None or
             0 <= current_source_s-last_seen <= .6)
    return TargetCandidate(entity["entity_id"], entity["label"], SelectionSource.DIRECT,
        float(entity.get("confidence", 0)), host_s, entity.get("last_seen_s"),
        tuple(entity["bounds_low_m"]), tuple(entity["bounds_high_m"]),
        world_frame, evidence={"entity": entity},
        track_state="FOUND" if fresh else "STALE")
