"""Uncertainty-aware arm geometry and sampled world intersections, not intent."""

from dataclasses import dataclass

import numpy as np

from .humans import JointState


@dataclass(frozen=True)
class PointingHypothesis:
    person_track_id: str
    timestamp_s: float
    arm: str | None
    origin: np.ndarray | None
    origin_covariance: np.ndarray | None
    direction: np.ndarray | None
    angular_uncertainty_rad: float | None  # RMS angle of sampled directions
    origins: np.ndarray
    directions: np.ndarray
    source_observations: tuple[str, ...]
    quality: float
    state: str
    reason: str


def _unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)


def estimate_pointing(track, now_s, samples=256, seed=4):
    def abstain(reason):
        return PointingHypothesis(track.person_id,now_s,None,None,None,None,None,
            np.empty((0,3)),np.empty((0,3)),track.pose.observation_ids,0.,"ABSTAIN",reason)
    if track.state!="VISIBLE" or now_s-track.last_seen_s>.25:
        return abstain("Person not freshly associated/observed")
    for side,quality in sorted(track.pose.hand_quality.items(),key=lambda v:-v[1]):
        base,tip = [track.pose.joints[f"{side}_index_{n}"] for n in ("mcp","tip")]
        if quality<.60 or any(j.position is None or now_s-j.timestamp_s>.20 for j in (base,tip)):
            continue
        length = np.linalg.norm(tip.position-base.position)
        if not .035<=length<=.16:
            continue
        direction = _unit(tip.position-base.position)
        rng = np.random.default_rng(seed)
        origins = rng.multivariate_normal(tip.position,tip.covariance,size=samples)
        bases = rng.multivariate_normal(base.position,base.covariance,size=samples)
        directions = _unit(origins-bases+rng.normal(0.,length*np.deg2rad(12.)/np.sqrt(2),(samples,3)))
        spread = float(np.sqrt(np.mean(np.arccos(np.clip(directions@direction,-1,1))**2)))
        return PointingHypothesis(track.person_id,now_s,side,tip.position,tip.covariance,direction,
            spread,origins,directions,track.pose.observation_ids,quality,"DEGRADED",
            "Estimated extended index with curled fingers; short baseline uncertainty")
    options = []
    for side in ("left","right"):
        s,e,w = [track.pose.joints[f"{side}_{n}"] for n in ("shoulder","elbow","wrist")]
        if any(j.position is None or now_s-j.timestamp_s>.25 for j in (s,w)):
            continue
        length = np.linalg.norm(w.position-s.position)
        if not .20<=length<=1.:
            continue
        direction = _unit(w.position-s.position)
        hip = track.pose.joints[f"{side}_hip"]
        down = _unit(hip.position-s.position) if hip.position is not None else np.array([0.,0.,-1.])
        # Continuous arm-extension and body-direction evidence. A downward resting
        # arm or folded arm does not become a confident target ray.
        resting_penalty = float(np.clip(1.-max(0.,np.dot(direction,down))**2,0.,1.))
        complete = e.position is not None and now_s-e.timestamp_s<=.25
        extension = 1.
        if complete:
            upper,lower = e.position-s.position,w.position-e.position
            extension = float(np.clip(length/(np.linalg.norm(upper)+np.linalg.norm(lower)),0.,1.))
            direction = _unit(.75*_unit(lower)+.25*direction)
        quality = min(s.confidence,w.confidence)*resting_penalty*extension**6
        if complete:
            quality *= e.confidence
        options.append((quality,side,s,e,w,direction,complete))
    if not options:
        return abstain("Insufficient fresh shoulder/wrist geometry")
    options.sort(key=lambda o:o[0],reverse=True)
    quality,side,s,e,w,direction,complete = options[0]
    if quality<.25:
        return abstain("Arm geometry does not support a pointing hypothesis")
    if len(options)>1 and options[1][0]>.7*quality:
        return abstain("Both arms plausible; arm choice unresolved")
    rng = np.random.default_rng(seed)
    def draw(j):
        return rng.multivariate_normal(j.position,j.covariance,size=samples)
    S,W = draw(s),draw(w)
    origins = W
    directions = _unit(W-S)
    if complete:
        E = draw(e)
        directions = _unit(.75*_unit(W-E)+.25*directions)
    cameras = set(s.source_cameras)|set(w.source_cameras)
    # Conservative model discrepancy floor; more independent views reduce it.
    floor = np.deg2rad(5. if len(cameras)==1 else 3.)
    if not complete:
        floor = np.deg2rad(14.)
    held = any(j.state==JointState.ESTIMATED for j in (s,w)) or complete and e.state==JointState.ESTIMATED
    if held:
        floor = max(floor,np.deg2rad(16.))
    if quality<.6:
        floor *= 1.5
    perturb = rng.normal(0.,floor/np.sqrt(2),(samples,3))
    perturb -= (perturb@direction)[:,None]*direction
    directions = _unit(directions+perturb)
    angles = np.arccos(np.clip(directions@direction,-1.,1.))
    uncertainty = float(np.sqrt(np.mean(angles**2)))
    degraded = not complete or held or len(cameras)==1 or quality<.6
    reason = ("Elbow unavailable; shoulder/wrist baseline" if not complete else
              "Held joint; uncertainty expanded" if held else
              "Single camera; model uncertainty floor" if len(cameras)==1 else "Multi-view arm geometry")
    return PointingHypothesis(track.person_id,now_s,side,w.position,w.covariance,direction,
        uncertainty,origins,directions,track.pose.observation_ids,quality,
        "DEGRADED" if degraded else "OK",reason)


def ray_box(origins,directions,low,high):
    # Slab test handles parallel rays without 0/0 at a box boundary.
    parallel = abs(directions)<1e-10
    outside = np.any(parallel & ((origins<low)|(origins>high)),axis=1)
    safe = np.where(parallel,1.,directions)
    a,b = (low-origins)/safe,(high-origins)/safe
    near = np.max(np.where(parallel,-np.inf,np.minimum(a,b)),axis=1)
    far = np.min(np.where(parallel,np.inf,np.maximum(a,b)),axis=1)
    t = np.maximum(near,0.)
    return np.where((far>=t)&~outside&(far>0.),t,np.inf)


def surface_hits(hypothesis,mapping,max_distance_m=6.):
    if mapping is None or not len(hypothesis.directions):
        return np.full(len(hypothesis.directions),np.inf),np.empty((0,3))
    cfg = mapping.config
    # Measured occupied voxels from M1, never analytic synthetic box surfaces.
    occupied = mapping.states()==2
    distances = np.full(len(hypothesis.directions),np.inf)
    # Ignore the immediate wrist/body voxel, whose occupied surface surrounds
    # the pointing origin. Near-field targets inside 20 cm are not supported.
    for step in np.arange(.20,max_distance_m,cfg.voxel_m/2):
        points = hypothesis.origins+hypothesis.directions*step
        indices = np.floor((points-np.asarray(cfg.origin))/cfg.voxel_m).astype(int)
        valid = np.all((indices>=0)&(indices<cfg.shape),axis=1)&~np.isfinite(distances)
        ids = np.flatnonzero(valid)
        if len(ids):
            hit = occupied[tuple(indices[ids].T)]
            distances[ids[hit]] = step
    finite = np.isfinite(distances)
    return distances,hypothesis.origins[finite]+hypothesis.directions[finite]*distances[finite,None]


def score_targets(hypothesis,entities,mapping=None,max_distance_m=6.):
    """Fraction of sampled first-hit entity bounds, including explicit unknown mass.

    These are geometric scores, not calibrated intent/target probabilities.
    M2 bounds contain observed surfaces; this is not exact object segmentation.
    """
    if hypothesis.state=="ABSTAIN":
        return {"scores": {"unknown":1.},"surface_points_m":[],"state":"ABSTAIN",
                "score_kind":"sampled geometric hit fraction"}
    surfaces,points = surface_hits(hypothesis,mapping,max_distance_m)
    best = np.full(len(hypothesis.directions),max_distance_m)
    winners = np.full(len(best),"unknown",dtype=object)
    ages = {e.entity_id:max(0.,hypothesis.timestamp_s-e.last_seen_s) for e in entities}
    for entity in entities:
        if ages[entity.entity_id]>.5:
            continue
        extent = np.sqrt(np.maximum(np.diag(entity.covariance),0.))
        padding = np.clip(extent,.015,.08)
        distances = ray_box(hypothesis.origins,hypothesis.directions,
                            entity.low-padding,entity.high+padding)
        valid = (distances<best)&(distances<=surfaces+.20)
        best[valid],winners[valid] = distances[valid],entity.entity_id
    scores = {e.entity_id:float(np.mean(winners==e.entity_id)) for e in entities}
    scores["unknown"] = float(np.mean(winners=="unknown"))
    return {"scores":dict(sorted(scores.items(),key=lambda item:-item[1])),
            "surface_points_m":points.tolist(),"state":hypothesis.state,
            "entity_data_age_s":ages,"stale_entities_excluded":[k for k,v in ages.items() if v>.5],
            "score_kind":"sampled geometric hit fraction"}
