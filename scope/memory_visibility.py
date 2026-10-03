"""Negative evidence requires visible, valid-depth rays through the old region."""

import numpy as np

from .data import RgbdFrame
from .mapping import FREE, VoxelMap


def visibility_check(geometry: dict, frame: RgbdFrame, T_global_episode: np.ndarray,
                     detector_reliability: float, alignment_sigma_m: float,
                     mapping: VoxelMap | None = None) -> dict:
    points = np.asarray(geometry["points_m"])
    pose = T_global_episode @ frame.T_world_camera
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    K = frame.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(camera[:, 0]*K.fx/z + K.cx)
        v = np.rint(camera[:, 1]*K.fy/z + K.cy)
    inside = (z>.15) & (z<5.) & (u>=0) & (u<K.width) & (v>=0) & (v<K.height)
    coverage = float(inside.mean())
    u, v = u[inside].astype(int), v[inside].astype(int)
    expected = z[inside]
    measured = frame.depth_m[v, u]
    valid = np.isfinite(measured) & (measured>=.15) & (measured<=5.)
    valid_fraction = float(valid.mean()) if len(valid) else 0.
    free = float(np.mean(measured[valid] > expected[valid]+.12)) if valid.any() else 0.
    occluded = float(np.mean(measured[valid] < expected[valid]-.10)) if valid.any() else 0.
    pixel_area = len(set(zip(u.tolist(), v.tolist())))
    map_free = None
    if mapping is not None:
        # Reuse M1's free/occupied/unknown classification in episode coordinates.
        local_points = (points-T_global_episode[:3, 3]) @ T_global_episode[:3, :3]
        ijk, valid_map = mapping._indices(local_points)
        map_free = float(np.mean(mapping.states()[tuple(ijk[valid_map].T)] == FREE)) \
            if valid_map.any() else 0.
    if coverage<.55 or pixel_area<12:
        result = "OUT_OF_FOV_OR_TOO_SMALL"
    elif valid_fraction<.80:
        result = "LOW_DEPTH_QUALITY"
    elif occluded>.20:
        result = "OCCLUDED"
    elif detector_reliability<.80:
        result = "DETECTOR_CAPABILITY_UNVERIFIED"
    elif alignment_sigma_m>.10:
        result = "ALIGNMENT_UNCERTAIN"
    elif free>=.65:
        result = "OBSERVED_ABSENT"
    else:
        result = "REGION_OCCUPIED_OR_INCONCLUSIVE"
    return {"frame_id": frame.frame_id, "time_s": frame.timestamp_s,
            "viewpoint_global": pose.tolist(), "expected_bounds_low_m": geometry["bounds_low_m"],
            "expected_bounds_high_m": geometry["bounds_high_m"],
            "coverage": coverage, "projected_unique_pixels": pixel_area,
            "valid_depth_fraction": valid_fraction, "free_ray_fraction": free,
            "occluded_fraction": occluded, "map_free_fraction": map_free,
            "detector_reliability": detector_reliability,
            "alignment_sigma_m": alignment_sigma_m, "result": result,
            "confidence": coverage*valid_fraction*free*detector_reliability
                          if result=="OBSERVED_ABSENT" else 0.}


def independent_negative_views(checks: list[dict]) -> list[dict]:
    selected = []
    for check in checks:
        if check["result"] != "OBSERVED_ABSENT":
            continue
        pose = np.asarray(check["viewpoint_global"])
        if all(np.linalg.norm(pose[:3,3]-np.asarray(old["viewpoint_global"])[:3,3])>=.10
               or np.arccos(np.clip(np.dot(pose[:3,2], np.asarray(old["viewpoint_global"])[:3,2]),-1,1))>=.10
               for old in selected):
            selected.append(check)
    return selected
