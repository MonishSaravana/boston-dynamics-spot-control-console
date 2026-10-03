"""Inspectable cross-episode matching with explicit abstention and alignment gates."""

from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .memory_store import MapAlignment


def aligned_entities(episode: dict) -> list[dict]:
    alignment = MapAlignment.from_dict(episode["alignment"])
    alignment.validate()
    R, t = alignment.T_global_episode[:3, :3], alignment.T_global_episode[:3, 3]
    result = []
    for local in episode["local_entities"]:
        state = dict(local)
        points = np.asarray(local["points_m"]) @ R.T + t
        corners = np.array(list(product(*zip(local["bounds_low_m"], local["bounds_high_m"]))))
        corners = corners @ R.T + t
        low, high = corners.min(axis=0), corners.max(axis=0)
        center = np.asarray(local["center_m"]) @ R.T + t
        sigma = alignment.sigma_at_global(center)
        state.update(center_m=center.tolist(), alignment_sigma_m=sigma,
                     bounds_low_m=low.tolist(), bounds_high_m=high.tolist(),
                     dimensions_m=(high-low).tolist(), points_m=points.tolist(),
                     covariance_m2=(R @ np.asarray(local["covariance_m2"]) @ R.T +
                                    np.eye(3) * (alignment.translation_sigma_m**2+
                                    (alignment.rotation_sigma_rad*np.linalg.norm(local["center_m"]))**2)).tolist())
        result.append(state)
    return result


def _context(entity: dict, others: list[dict]) -> dict[str, float]:
    result = {}
    for other in others:
        if other["entity_id"] == entity["entity_id"]:
            continue
        label = other["label"]
        distance = float(np.linalg.norm(np.array(other["center_m"]) - entity["center_m"]))
        result[label] = min(result.get(label, float("inf")), distance)
    return result


def identity_candidates(episode: dict, beliefs: dict) -> tuple[list[dict], list[list[dict]]]:
    if any(b["alignment"]["world_frame"]!=episode["alignment"]["world_frame"] for b in beliefs.values()):
        raise ValueError("Identity scoring requires the same declared global coordinate frame")
    local = aligned_entities(episode)
    globals_ = list(beliefs.values())
    # A static-class contradiction is diagnostic, not an estimated registration.
    anchors = [float(np.linalg.norm(np.array(a["center_m"]) - b["geometry"]["center_m"]))
               for a in local for b in globals_ if a["label"] == b["label"] == "table"]
    residual = min(anchors) if anchors else 0.
    alignment = episode["alignment"]
    candidates = []
    for entity in local:
        row = []
        for old in globals_:
            geometry = old["geometry"]
            affinity = sum(p * old["class_probabilities"].get(k, 0.)
                           for k, p in entity["class_probabilities"].items())
            distance = float(np.linalg.norm(np.array(entity["center_m"]) - geometry["center_m"]))
            a, b = np.maximum(entity["dimensions_m"], .05), np.maximum(geometry["dimensions_m"], .05)
            dimension_score = float(np.exp(-np.mean(np.abs(np.log(a/b))) / .55))
            new_points = np.asarray(entity["points_m"]) - entity["center_m"]
            old_points = np.asarray(geometry["points_m"]) - geometry["center_m"]
            surface_error = float((np.median(cKDTree(old_points).query(new_points)[0]) +
                                   np.median(cKDTree(new_points).query(old_points)[0])) / 2)
            surface_score = float(np.exp(-surface_error / .16))
            mobility = {"table": .10, "chair": .75, "backpack": 1.0}.get(entity["label"], .5)
            elapsed = max(0., episode["start_s"] - old["last_positive"]["time_s"])
            location_scale = .18 + mobility * min(1.6, .5 + elapsed / 3600.)
            location_score = float(np.exp(-.5 * (distance/location_scale)**2))
            old_context = old.get("context", {})
            new_context = _context(entity, local)
            common = set(old_context) & set(new_context)
            context_score = float(np.exp(-np.mean([abs(new_context[k]-old_context[k])
                                                   for k in common]) / .8)) if common else .5
            # Both the old and the new episode alignment affect correspondence.
            old_sigma = MapAlignment.from_dict(old["alignment"]).sigma_at_global(geometry["center_m"])
            sigma = float(np.hypot(entity["alignment_sigma_m"], old_sigma))
            alignment_quality = float(np.exp(-sigma/.5 - residual/.7))
            score = float((.25*affinity + .22*dimension_score + .18*surface_score +
                           .25*location_score + .10*context_score) * alignment_quality)
            reasons = []
            if affinity < .25:
                reasons.append("class incompatibility")
            if dimension_score < .38 or surface_score < .30:
                reasons.append("shape incompatibility")
            if distance > .35 + 3.0*mobility:
                reasons.append("movement exceeds mobility gate")
            if sigma > .20 or residual > .50:
                reasons.append("alignment uncertainty or static-anchor contradiction")
            if score < .66:
                reasons.append("insufficient combined evidence")
            row.append({"local_id": entity["entity_id"], "global_id": old["global_id"],
                        "score": score, "class_affinity": affinity,
                        "dimension_score": dimension_score, "surface_shape_score": surface_score,
                        "centered_surface_error_m": surface_error, "distance_m": distance,
                        "location_score": location_score, "context_score": context_score,
                        "mobility_prior": mobility, "elapsed_s": elapsed,
                        "alignment_sigma_m": sigma, "static_anchor_residual_m": residual,
                        "alignment_quality": alignment_quality,
                        "accepted": not reasons, "reasons": reasons})
        candidates.append(sorted(row, key=lambda c: c["score"], reverse=True))
    return local, candidates


def match_episode(episode: dict, beliefs: dict) -> list[dict]:
    """Confident unique links first; ambiguous alternatives remain explicit."""
    local, candidates = identity_candidates(episode, beliefs)
    remaining = set(range(len(local)))
    used = set()
    decisions = {}
    while remaining:
        proposals = []
        for index in sorted(remaining):
            viable = [c for c in candidates[index] if c["accepted"] and c["global_id"] not in used]
            if not viable:
                continue
            best = viable[0]
            margin = best["score"] - viable[1]["score"] if len(viable)>1 else 1.
            competition = [c["score"] for j in remaining if j != index
                           for c in candidates[j] if c["global_id"] == best["global_id"] and c["accepted"]]
            global_margin = best["score"] - max(competition) if competition else 1.
            if margin >= .085 and global_margin >= .085:
                proposals.append((best["score"], index, best, margin, global_margin))
        if not proposals:
            break
        _, index, best, margin, global_margin = max(proposals, key=lambda x:x[0])
        decisions[index] = {"local_id": local[index]["entity_id"], "global_id": best["global_id"],
                            "status": "LINKED", "score": best["score"],
                            "margin": margin, "global_margin": global_margin,
                            "candidates": candidates[index]}
        used.add(best["global_id"])
        remaining.remove(index)
    for index in remaining:
        row = candidates[index]
        # Partial surfaces can look unlike their old full view. A compatible
        # unused identity warrants abstention rather than an invented new ID.
        plausible = [c for c in row if c["class_affinity"]>=.25 and c["global_id"] not in used]
        status = "UNRESOLVED" if plausible else "NEW"
        decisions[index] = {"local_id": local[index]["entity_id"], "global_id": None,
                            "status": status, "score": row[0]["score"] if row else None,
                            "reason": "ambiguous or insufficient identity evidence" if plausible else
                                      "no remaining compatible global hypothesis",
                            "candidates": row}
    return [decisions[i] for i in range(len(local))]


def entity_context(entity, episode_entities):
    return _context(entity, episode_entities)
