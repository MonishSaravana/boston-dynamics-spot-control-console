"""Reproducible Rerun recording with synchronized images, geometry and states."""

from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np

from .data import RgbdFrame, camera_to_world, project_depth
from .mapping import FREE, OCCUPIED, UNKNOWN, MapConfig, VoxelMap


def _state_image(mapping: VoxelMap) -> np.ndarray:
    state = mapping.topdown().T[::-1]
    image = np.empty(state.shape + (3,), dtype=np.uint8)
    image[state == UNKNOWN] = [57, 64, 76]
    image[state == FREE] = [65, 153, 166]
    image[state == OCCUPIED] = [235, 156, 74]
    return image


def record_viewer(frames: list[RgbdFrame], config: MapConfig, output: Path,
                  mesh=None) -> VoxelMap:
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Growing 3D reconstruction",
                              contents=["world/**"], background=[17, 25, 35]),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/rgb", name="RGB observation"),
                rrb.Spatial2DView(origin="world/camera/depth", name="Depth in meters"),
                rrb.Spatial2DView(origin="map/topdown",
                                  name="Occupied orange · Free teal · Unknown gray"),
            ), column_shares=[2.2, 1.0]),
        rrb.BlueprintPanel(expanded=False),
        rrb.SelectionPanel(expanded=False),
        rrb.TimePanel(expanded=True, timeline="frame"),
        auto_layout=False, auto_views=False)
    rr.init("SCOPE known-pose RGB-D mapping")
    rr.save(output / "reconstruction.rrd", default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/axes", rr.TransformAxes3D(axis_length=.5), static=True)
    rr.log("legend", rr.TextLog(
        "Map: orange = occupied, teal = observed free, dark gray = unknown. "
        "RGB-D and poses are simulated or measured as labeled in the run manifest."), static=True)

    mapping = VoxelMap(config)
    trajectory = []
    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)
        rr.set_time("elapsed", duration=frame.timestamp_s - frames[0].timestamp_s)
        accepted = mapping.integrate(frame)
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
        trajectory.append(frame.T_world_camera[:3, 3].copy())
        rr.log("world/trajectory", rr.LineStrips3D([trajectory], colors=[35, 218, 219],
                                                     radii=.02))
        current, colors, _ = project_depth(frame, stride=6)
        if len(current):
            rr.log("world/current_projected_depth", rr.Points3D(
                camera_to_world(current, frame.T_world_camera), colors=colors, radii=.012))
        if accepted:
            points, cloud_colors, counts = mapping.cloud()
            rr.log("world/integrated_surface_points", rr.Points3D(
                points, colors=cloud_colors, radii=np.clip(.008 + .002 * counts, .008, .025)))
        rr.log("map/topdown", rr.Image(_state_image(mapping)))
        rr.log("status", rr.TextLog(
            f"frame {i+1}/{len(frames)} | {frame.source} | "
            f"{mapping.revision} integrated | {len(mapping.rejected)} rejected"))
        if not accepted:
            rr.log("errors", rr.TextLog(mapping.rejected[-1], level=rr.TextLogLevel.ERROR))
        if mapping.warnings and mapping.warnings[-1].startswith(frame.frame_id):
            rr.log("warnings", rr.TextLog(mapping.warnings[-1], level=rr.TextLogLevel.WARN))
    if mesh is not None and len(mesh.vertices):
        rr.set_time("frame", sequence=len(frames)-1)
        rr.log("world/tsdf_surface", rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.triangles,
            vertex_colors=np.tile(np.array([[159, 208, 232]], dtype=np.uint8),
                                  (len(mesh.vertices), 1))))
    return mapping


def open_recording(path: Path) -> subprocess.Popen:
    """Launch the native interactive viewer; the recording remains usable later."""
    executable = Path(sys.executable).with_name("rerun")
    if not executable.exists():
        found = shutil.which("rerun")
        if found is None:
            raise RuntimeError("Rerun viewer was not installed with rerun-sdk")
        executable = Path(found)
    return subprocess.Popen([str(executable), str(path.resolve())],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
