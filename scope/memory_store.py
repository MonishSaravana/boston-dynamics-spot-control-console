"""Append-only episode evidence and belief revisions; RGB-D stays in run assets."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from .data import RgbdFrame
from .semantic_pipeline import SemanticRun

ALGORITHM = "scope-memory-v1"


@dataclass(frozen=True)
class MapAlignment:
    """A supplied rigid map-to-global transform with declared uncertainty."""
    world_frame: str
    provenance: str
    T_global_episode: np.ndarray = field(default_factory=lambda: np.eye(4))
    translation_sigma_m: float = 0.
    rotation_sigma_rad: float = 0.

    def validate(self):
        if not self.world_frame or self.provenance not in ("known_shared", "external_rigid"):
            raise ValueError("Alignment requires a named world frame and explicit provenance")
        T = self.T_global_episode
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("Invalid alignment transform")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError("Alignment rotation must be rigid")
        if not np.isfinite([self.translation_sigma_m, self.rotation_sigma_rad]).all() or min(
                self.translation_sigma_m, self.rotation_sigma_rad) < 0:
            raise ValueError("Alignment uncertainty must be finite and nonnegative")

    def summary(self):
        self.validate()
        return {**asdict(self), "T_global_episode": self.T_global_episode.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["world_frame"], data["provenance"],
                   np.asarray(data["T_global_episode"], dtype=float),
                   data.get("translation_sigma_m", 0.), data.get("rotation_sigma_rad", 0.))


def asset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_snapshot(episode_id: str, directory: Path, run: SemanticRun,
                     alignment: MapAlignment, *, time_offset_s: float = 0.,
                     time_domain: str = "source_seconds") -> dict:
    """Detach local hypotheses and their evidence before any global processing."""
    alignment.validate()
    if not np.isfinite(time_offset_s) or time_domain not in ("source_seconds", "unix_utc"):
        raise ValueError("Explicit finite clock offset and supported time domain required")
    if run.store is None or not run.frames:
        raise ValueError("Memory requires a nonempty entity-stage episode")
    local = []
    for entity in run.store.entities.values():
        state = entity.summary()
        state["points_m"] = entity.points_world[
            ::max(1, int(np.ceil(len(entity.points_world) / 1200)))].tolist()
        state["frame_ids"] = sorted(entity.supporting_views)
        state["first_seen_s"] += time_offset_s
        state["last_seen_s"] += time_offset_s
        local.append(state)
    assets = {}
    for name in ("episode.npz", "manifest.json", "map.npz", "map_config.json",
                 "semantic_masks.npz", "detections.json", "entities.json", "association.json",
                 "object_observations_3d.json", "semantic_manifest.json"):
        path = directory / name
        if path.exists():
            assets[name] = asset_digest(path)
    if not {"episode.npz", "detections.json", "semantic_masks.npz"}.issubset(assets):
        raise ValueError("Save the RGB-D episode and semantic evidence before importing memory")
    manifest = directory / "semantic_manifest.json"
    detector = json.loads(manifest.read_text()) if manifest.exists() else {
        "detector": run.detector_name, "version": "legacy-m2-saved-masks"}
    return {
        "episode_id": episode_id, "start_s": run.frames[0].timestamp_s+time_offset_s,
        "end_s": run.frames[-1].timestamp_s+time_offset_s, "asset_directory": str(directory.resolve()),
        "clock_offset_s": time_offset_s, "time_domain": time_domain,
        "asset_sha256": assets, "alignment": alignment.summary(),
        "map_reference": "map.npz" if "map.npz" in assets else None,
        "map_revision": len(run.frames), "detector": detector,
        "local_entities": local, "local_association": run.store.decisions,
        "observations": [{"observation_id": obs.observation_id,
                          "frame_id": obs.detection.frame_id, "time_s": obs.timestamp_s+time_offset_s,
                          "valid_depth_fraction": obs.valid_depth_fraction}
                         for group in run.projections for obs in group],
        "frames": [{"frame_id": f.frame_id, "time_s": f.timestamp_s+time_offset_s,
                    "source_time_s": f.timestamp_s,
                    "T_episode_camera": f.T_world_camera.tolist(),
                    "intrinsics": asdict(f.intrinsics), "asset_index": index}
                   for index, f in enumerate(run.frames)],
        "algorithm": ALGORITHM,
    }


class MemoryStore:
    """Every database row is immutable; current belief is the latest revision."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys=ON")
        schema = {
            "episodes": "episode_id TEXT PRIMARY KEY, start_s REAL, end_s REAL, payload TEXT",
            "links": "seq INTEGER PRIMARY KEY, episode_id TEXT REFERENCES episodes, local_id TEXT, global_id TEXT, status TEXT, payload TEXT, UNIQUE(episode_id,local_id)",
            "beliefs": "seq INTEGER PRIMARY KEY, episode_id TEXT REFERENCES episodes, global_id TEXT, time_s REAL, payload TEXT",
            "events": "seq INTEGER PRIMARY KEY, episode_id TEXT REFERENCES episodes, global_id TEXT, time_s REAL, kind TEXT, payload TEXT",
            "processed": "episode_id TEXT PRIMARY KEY REFERENCES episodes, algorithm TEXT",
        }
        for name, columns in schema.items():
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {name} ({columns})")
            for operation in ("UPDATE", "DELETE"):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {name}_no_{operation.lower()} "
                                f"BEFORE {operation} ON {name} BEGIN "
                                "SELECT RAISE(ABORT, 'Memory history is append-only'); END")
        self.db.commit()

    def close(self):
        self.db.close()

    def import_snapshot(self, snapshot: dict):
        episodes = self.episodes()
        if episodes and snapshot["alignment"]["world_frame"] != episodes[0]["alignment"]["world_frame"]:
            raise ValueError("Episodes must name the same global coordinate frame")
        if episodes and snapshot["time_domain"] != episodes[0]["time_domain"]:
            raise ValueError("Episodes must declare a common time domain")
        encoded = json.dumps(snapshot, sort_keys=True)
        old = self.db.execute("SELECT payload FROM episodes WHERE episode_id=?",
                              (snapshot["episode_id"],)).fetchone()
        if old:
            if old[0] != encoded:
                raise ValueError("Episode ID already has different immutable evidence")
            return
        with self.db:
            self.db.execute("INSERT INTO episodes VALUES (?,?,?,?)", (
                snapshot["episode_id"], snapshot["start_s"], snapshot["end_s"], encoded))

    def episodes(self):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM episodes ORDER BY start_s,episode_id")]

    def episode(self, episode_id):
        row = self.db.execute("SELECT payload FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown episode {episode_id}")
        return json.loads(row[0])

    def verify_assets(self, episode_id):
        episode = self.episode(episode_id)
        directory = Path(episode["asset_directory"])
        for name, expected in episode["asset_sha256"].items():
            if not (directory / name).exists() or asset_digest(directory / name) != expected:
                raise ValueError(f"Episode evidence changed or disappeared: {directory / name}")
        return directory

    def beliefs(self, through_episode: str | None = None):
        limit = self.episode(through_episode)["end_s"] if through_episode else float("inf")
        result = {}
        for global_id, payload in self.db.execute(
                "SELECT global_id,payload FROM beliefs WHERE time_s<=? ORDER BY seq", (limit,)):
            result[global_id] = json.loads(payload)
        return result

    def links(self, episode_id=None):
        query = "SELECT payload FROM links"
        args = ()
        if episode_id:
            query += " WHERE episode_id=?"
            args = (episode_id,)
        return [json.loads(row[0]) for row in self.db.execute(query + " ORDER BY seq", args)]

    def history(self, global_id=None):
        query = "SELECT seq,episode_id,global_id,time_s,kind,payload FROM events"
        args = ()
        if global_id:
            query += " WHERE global_id=?"
            args = (global_id,)
        return [{"seq": row[0], "episode_id": row[1], "global_id": row[2],
                 "time_s": row[3], "kind": row[4], **json.loads(row[5])}
                for row in self.db.execute(query + " ORDER BY time_s,seq", args)]

    def append_event(self, episode_id, global_id, time_s, kind, evidence):
        self.db.execute("INSERT INTO events(episode_id,global_id,time_s,kind,payload) VALUES (?,?,?,?,?)",
                        (episode_id, global_id, time_s, kind,
                         json.dumps({"algorithm": ALGORITHM, **evidence})))

    def append_belief(self, episode_id, global_id, time_s, belief):
        self.db.execute("INSERT INTO beliefs(episode_id,global_id,time_s,payload) VALUES (?,?,?,?)",
                        (episode_id, global_id, time_s, json.dumps(belief)))

    def last_seen(self, global_id):
        belief = self.beliefs().get(global_id)
        if belief is None:
            raise ValueError(f"Unknown global entity {global_id}")
        positive = belief["last_positive"]
        directory = self.verify_assets(positive["episode_id"])
        frame_ids = {f["frame_id"] for f in self.episode(positive["episode_id"])["frames"]}
        if positive["frame_id"] not in frame_ids:
            raise ValueError("Last-seen evidence references a missing frame")
        return {"global_id": global_id, "status": belief["status"],
                "location_confidence": belief["location_confidence"],
                "last_positive": positive, "evidence_directory": str(directory),
                "later_negative_checks": sum(event["kind"] == "NOT_VISIBLE_FROM_VIEW" and
                                             event["time_s"] > positive["time_s"]
                                             for event in self.history(global_id)),
                "unobserved_since_s": belief.get("unobserved_since_s"),
                "age_at_latest_episode_s": max(e["end_s"] for e in self.episodes())-positive["time_s"],
                "time_domain": self.episode(positive["episode_id"])["time_domain"],
                "rerun_reference": str(directory / "reconstruction.rrd")}
