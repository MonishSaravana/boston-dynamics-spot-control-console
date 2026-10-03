"""Global hypotheses and explicit events over immutable aligned episodes."""

import copy
from dataclasses import replace
import json

import numpy as np

from .memory_identity import aligned_entities, entity_context, match_episode
from .memory_store import ALGORITHM, MemoryStore
from .memory_visibility import independent_negative_views, visibility_check


def _positive(episode, entity):
    supporting = [frame for frame in episode["frames"] if frame["frame_id"] in entity["frame_ids"]]
    frame = max(supporting, key=lambda f:f["time_s"])
    return {"episode_id": episode["episode_id"], "local_id": entity["entity_id"],
            "time_s": frame["time_s"], "frame_id": frame["frame_id"],
            "asset_index": frame["asset_index"], "position_m": entity["center_m"],
            "position_kind": "episode-fused observed-bounds center",
            "observation_ids": entity["observation_ids"],
            "map_revision": episode["map_revision"], "map_reference": episode["map_reference"]}


def process_episode(store: MemoryStore, episode_id: str) -> list[dict]:
    if store.db.execute("SELECT 1 FROM processed WHERE episode_id=?", (episode_id,)).fetchone():
        return store.links(episode_id)
    episode = store.episode(episode_id)
    store.verify_assets(episode_id)
    latest = store.db.execute("SELECT MAX(e.end_s) FROM episodes e JOIN processed p USING(episode_id)").fetchone()[0]
    if latest is not None and episode["start_s"] <= latest:
        raise ValueError("Process nonoverlapping episodes in chronological order; history cannot be rewritten")
    prior = store.beliefs()
    decisions = match_episode(episode, prior)
    locals_ = aligned_entities(episode)
    by_id = {e["entity_id"]: e for e in locals_}
    updated = copy.deepcopy(prior)
    seen, uncertain = set(), set()
    time = episode["end_s"]
    # Real model miss probability is unmeasured; default to no absence claim.
    reliability = episode["detector"].get("negative_detection_reliability",
                   1. if episode["detector"]["detector"]=="synthetic-truth" else 0.)
    with store.db:
        for decision in decisions:
            entity = by_id[decision["local_id"]]
            evidence_ids = [f"{episode_id}/{oid}" for oid in entity["observation_ids"]]
            if decision["status"] == "UNRESOLVED":
                compatible = [c for c in decision["candidates"] if c["class_affinity"]>=.25]
                for candidate in compatible:
                    gid = candidate["global_id"]
                    if gid in seen:
                        continue
                    uncertain.add(gid)
                    updated[gid]["status"] = "IDENTITY_UNCERTAIN"
                    store.append_event(episode_id, gid, time, "IDENTITY_UNCERTAIN", {
                        "local_id": entity["entity_id"], "evidence_ids": evidence_ids,
                        "confidence": candidate["score"], "identity_audit": decision})
            else:
                gid = decision["global_id"]
                if gid is None:
                    label = entity["label"].replace(" ", "_")
                    index = 1
                    while f"global/{label}_{index:04d}" in updated:
                        index += 1
                    gid = f"global/{label}_{index:04d}"
                    decision["global_id"] = gid
                    store.append_event(episode_id, gid, time, "NEW_ENTITY", {
                        "confidence": entity["confidence"], "evidence_ids": evidence_ids})
                old = prior.get(gid)
                probabilities = dict(entity["class_probabilities"])
                if old:
                    labels = set(probabilities) | set(old["class_probabilities"])
                    probabilities = {k:(probabilities.get(k,0.)+old["class_probabilities"].get(k,0.))/2
                                     for k in labels}
                confidence = entity["confidence"] * (decision["score"] if old else 1.)
                belief = {"global_id": gid, "label": max(probabilities, key=probabilities.get),
                          "class_probabilities": probabilities, "geometry": entity,
                          "status": "VISIBLE", "location_confidence": confidence,
                          "last_positive": _positive(episode, entity),
                          "context": entity_context(entity, locals_),
                          "alignment": episode["alignment"], "unobserved_since_s": None,
                          "location_supported": True,
                          "algorithm": ALGORITHM}
                if old:
                    distance = float(np.linalg.norm(np.array(entity["center_m"])-old["geometry"]["center_m"]))
                    sigma = episode["alignment"]["translation_sigma_m"]
                    if distance > max(.20, 3*sigma):
                        certain = decision["score"]>=.72 and decision["margin"]>=.085
                        kind = "MOVED" if certain else "POSSIBLY_MOVED"
                        if not certain:
                            belief["status"] = "POSSIBLY_MOVED"
                        store.append_event(episode_id, gid, time, kind, {
                            "previous_episode": old["last_positive"]["episode_id"],
                            "from_m": old["geometry"]["center_m"], "to_m": entity["center_m"],
                            "distance_m": distance, "confidence": decision["score"],
                            "evidence_ids": evidence_ids, "identity_audit": decision})
                store.append_event(episode_id, gid, time, "IDENTITY_LINK", {
                    "local_id": entity["entity_id"], "identity_audit": decision,
                    "confidence": decision["score"], "evidence_ids": evidence_ids})
                store.append_event(episode_id, gid, belief["last_positive"]["time_s"],
                                   "REOBSERVED" if old else "OBSERVED", {
                                       "position_m": entity["center_m"], "confidence": confidence,
                                       "evidence_ids": evidence_ids, "positive": belief["last_positive"]})
                updated[gid] = belief
                seen.add(gid)
            store.db.execute("INSERT INTO links(episode_id,local_id,global_id,status,payload) VALUES (?,?,?,?,?)",
                             (episode_id, entity["entity_id"], decision["global_id"],
                              decision["status"], json.dumps(decision)))
        for gid in prior.keys() - seen - uncertain:
            belief = updated[gid]
            belief["status"] = "NOT_CURRENTLY_OBSERVED"
            belief["unobserved_since_s"] = belief.get("unobserved_since_s") or episode["start_s"]
            store.append_event(episode_id, gid, time, "NOT_OBSERVED", {
                "previous_position_m": belief["geometry"]["center_m"],
                "confidence": belief["location_confidence"], "evidence_ids": [],
                "reason": "No linked positive observation in this episode"})
            from pathlib import Path
            from .storage import load_episode, load_map
            directory = Path(episode["asset_directory"])
            frames = load_episode(directory)
            offset = episode["clock_offset_s"]
            frames = [replace(f, timestamp_s=f.timestamp_s+offset,
                              depth_timestamp_s=f.depth_timestamp_s+offset) for f in frames]
            mapping = load_map(directory) if episode["map_reference"] else None
            alignment = episode["alignment"]
            transform = np.asarray(alignment["T_global_episode"])
            sigma = alignment["translation_sigma_m"] + alignment["rotation_sigma_rad"] * \
                float(np.linalg.norm(belief["geometry"]["center_m"]))
            checks = [visibility_check(belief["geometry"], f, transform, reliability, sigma, mapping)
                      for f in frames]
            for index, check in enumerate(checks):
                check["asset_index"] = index
                check["evidence_ids"] = [f"{episode_id}/frame/{check['frame_id']}"]
                store.append_event(episode_id, gid, check["time_s"],
                                   "NOT_VISIBLE_FROM_VIEW" if check["result"]=="OBSERVED_ABSENT"
                                   else "VISIBILITY_CHECK", check)
            independent = independent_negative_views(checks)
            if len(independent)>=2:
                belief["status"] = "POSSIBLY_MISSING"
                belief["location_supported"] = False
                belief["location_confidence"] *= .35
                store.append_event(episode_id, gid, time, "MISSING_HYPOTHESIS", {
                    "previous_status": prior[gid]["status"], "new_status": belief["status"],
                    "independent_negative_views": len(independent),
                    "evidence_ids": [key for check in independent for key in check["evidence_ids"]],
                    "confidence": float(np.mean([c["confidence"] for c in independent])),
                    "last_positive_position_m": belief["geometry"]["center_m"]})
        for gid, belief in updated.items():
            store.append_belief(episode_id, gid, time, belief)
        store.db.execute("INSERT INTO processed VALUES (?,?)", (episode_id, ALGORITHM))
    return decisions
