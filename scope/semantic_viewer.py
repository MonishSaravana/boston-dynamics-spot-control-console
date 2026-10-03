"""Rerun overlays for masks, projected points, entities, and provenance."""

from pathlib import Path
import json

import numpy as np

from .data import RgbdFrame
from .entities import EntityStore
from .semantic_pipeline import SemanticRun


_PALETTE = np.array([
    [68, 207, 220], [245, 171, 82], [235, 105, 137],
    [142, 191, 110], [178, 144, 229], [241, 215, 103],
], dtype=np.uint8)


def _color(index: int) -> np.ndarray:
    return _PALETTE[index % len(_PALETTE)]


class SemanticRecorder:
    def __init__(self, run: SemanticRun):
        self.run = run
        self.store = EntityStore() if run.stage == "entities" else None

    def log_frame(self, frame: RgbdFrame, index: int) -> None:
        import rerun as rr

        detections = self.run.detections[index]
        projected = self.run.projections[index]
        overlay = frame.rgb.astype(np.float32).copy()
        boxes, sizes, labels, colors = [], [], [], []
        for n, detection in enumerate(detections):
            color = _color(n)
            overlay[detection.mask] = .50 * overlay[detection.mask] + .50 * color
            x1, y1, x2, y2 = detection.box_xyxy
            boxes.append([x1, y1])
            sizes.append([x2 - x1, y2 - y1])
            labels.append(f"{detection.label} {detection.confidence:.0%}")
            colors.append(color)
        rr.log("world/camera/semantic_masks", rr.Image(overlay.astype(np.uint8)))
        if boxes:
            rr.log("world/camera/rgb/detections", rr.Boxes2D(
                mins=boxes, sizes=sizes, labels=labels, colors=colors,
                show_labels=True, radii=2.0))
        else:
            rr.log("world/camera/rgb/detections", rr.Clear(recursive=True))
        if projected:
            positions = np.concatenate([item.points_world[::max(1, len(item.points_world)//500)]
                                        for item in projected])
            point_colors = np.concatenate([
                np.tile(_color(n), (len(item.points_world[::max(1, len(item.points_world)//500)]), 1))
                for n, item in enumerate(projected)])
            rr.log("world/current_object_points", rr.Points3D(
                positions, colors=point_colors, radii=.018))
        else:
            rr.log("world/current_object_points", rr.Clear(recursive=True))
        if self.store is None:
            return
        self.store.add_frame(projected)
        for n, entity in enumerate(self.store.entities.values()):
            path = f"world/entities/{entity.entity_id}"
            rr.log(path, rr.Boxes3D(
                centers=[entity.center], sizes=[entity.high - entity.low],
                colors=[_color(n)], radii=.018, show_labels=False,
                fill_mode=rr.components.FillMode.MajorWireframe))
            label_position = entity.center.copy()
            label_position[2] = entity.high[2] + .12 + .055 * n
            rr.log(path + "/label", rr.Points3D(
                [label_position], labels=[f"{entity.entity_id} [{entity.confidence:.0%}]"],
                colors=[_color(n)], radii=.025, show_labels=True))
            supporting = [decision for decision in self.store.decisions
                          if decision["selected"]["entity_id"] == entity.entity_id]
            evidence = {"entity": entity.summary(), "association_decisions": supporting}
            rr.log(path + "/evidence", rr.TextDocument(
                json.dumps(evidence, indent=2), media_type="application/json"))
        rr.log("semantic/status", rr.TextLog(
            f"{len(detections)} masks | {len(projected)} projected | "
            f"{len(self.store.entities)} entities"))


def record_semantic_only(run: SemanticRun, output: Path) -> EntityStore | None:
    import rerun as rr
    import rerun.blueprint as rrb

    if run.stage == "detect":
        layout = rrb.Horizontal(
            rrb.Spatial2DView(origin="world/camera/rgb", name="RGB and boxes"),
            rrb.Spatial2DView(origin="world/camera/semantic_masks", name="Instance masks"),
            column_shares=[1, 1])
    else:
        centers = np.array([item.center for group in run.projections for item in group])
        target = (centers.min(axis=0) + centers.max(axis=0)) / 2 if len(centers) else np.zeros(3)
        span = max(1.0, float(np.max(np.ptp(centers, axis=0)))) if len(centers) else 2.0
        layout = rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Projected objects and entities",
                              contents=["world/**"], background=[17, 25, 35],
                              line_grid=False,
                              eye_controls=rrb.EyeControls3D(
                                  position=target + np.array([1.4, -2.0, 1.1]) * span,
                                  look_target=target, eye_up=[0, 0, 1])),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/rgb", name="RGB and boxes"),
                rrb.Spatial2DView(origin="world/camera/semantic_masks", name="Instance masks"),
                rrb.Spatial2DView(origin="world/camera/depth", name="Depth in meters")),
            column_shares=[2.2, 1.0])
    blueprint = rrb.Blueprint(layout, rrb.BlueprintPanel(expanded=False),
                              rrb.SelectionPanel(expanded=False),
                              rrb.TimePanel(expanded=True, timeline="frame"),
                              auto_layout=False, auto_views=False)
    rr.init(f"SCOPE semantic objects: {run.stage}")
    rr.save(output / "semantic.rrd", default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/axes", rr.TransformAxes3D(axis_length=.5), static=True)
    recorder = SemanticRecorder(run)
    for index, frame in enumerate(run.frames):
        rr.set_time("frame", sequence=index)
        rr.set_time("elapsed", duration=frame.timestamp_s - run.frames[0].timestamp_s)
        rr.log("world/camera", rr.Transform3D(
            translation=frame.T_world_camera[:3, 3],
            mat3x3=frame.T_world_camera[:3, :3]))
        K = frame.intrinsics
        rr.log("world/camera", rr.Pinhole(
            width=K.width, height=K.height, focal_length=[K.fx, K.fy],
            principal_point=[K.cx, K.cy], camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=.35))
        rr.log("world/camera/rgb", rr.Image(frame.rgb))
        rr.log("world/camera/depth", rr.DepthImage(
            np.nan_to_num(frame.depth_m, nan=0.), meter=1.0, depth_range=[0, 5]))
        recorder.log_frame(frame, index)
    rr.disconnect()
    return recorder.store
