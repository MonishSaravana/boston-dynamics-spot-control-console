"""Deterministic geometry-only relations between current episode entities."""

import numpy as np

from .entities import WorldEntity


def entity_relations(entities: list[WorldEntity], near_m: float = 1.25) -> list[dict]:
    """World-X ordering is coordinate-frame ordering, not a viewer's left/right."""
    result = []
    for source in entities:
        for target in entities:
            if source.entity_id == target.entity_id:
                continue
            distance = float(np.linalg.norm(source.center - target.center))
            names = []
            if distance <= near_m:
                names.append("near")
            if source.center[0] < target.center[0] - .05:
                names.append("left_of_world_x")
            elif source.center[0] > target.center[0] + .05:
                names.append("right_of_world_x")
            if source.low[2] > target.high[2] + .05:
                names.append("above")
            elif source.high[2] < target.low[2] - .05:
                names.append("below")
            xy_overlap = np.maximum(0, np.minimum(source.high[:2], target.high[:2]) -
                                    np.maximum(source.low[:2], target.low[:2]))
            xy_area = float(np.prod(xy_overlap))
            source_area = float(np.prod(np.maximum(source.high[:2] - source.low[:2], .01)))
            vertical_gap = float(source.low[2] - target.high[2])
            if xy_area / source_area >= .3 and -.04 <= vertical_gap <= .08:
                names.append("supported_by")
            result.append({"source": source.entity_id, "target": target.entity_id,
                           "distance_m": distance, "relations": names})
    return result
