"""Rerun episode-end beliefs, retained historical positions, and event evidence."""

import json
from pathlib import Path

import numpy as np

from .data import camera_to_world
from .memory_store import MemoryStore
from .semantic_storage import load_semantic_run
from .storage import load_map


def record_memory(store: MemoryStore, output: Path) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb
    output.mkdir(parents=True, exist_ok=True)
    episodes = [e for e in store.episodes() if store.db.execute(
        "SELECT 1 FROM processed WHERE episode_id=?", (e["episode_id"],)).fetchone()]
    if not episodes:
        raise ValueError("Build at least one memory episode before replay")
    all_centers = np.array([b["geometry"]["center_m"] for e in episodes
                           for b in store.beliefs(e["episode_id"]).values()])
    target = (all_centers.min(axis=0)+all_centers.max(axis=0))/2 if len(all_centers) else np.zeros(3)
    span = max(1.5, float(np.max(np.ptp(all_centers, axis=0)))) if len(all_centers) else 2.
    layout = rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="Current belief and retained history",
                          contents=["world/**", "- /world/camera/depth/**"],
                          background=[17,25,35], line_grid=False,
                          eye_controls=rrb.EyeControls3D(position=target+np.array([1.4,-2.,1.25])*span,
                                                       look_target=target, eye_up=[0,0,1])),
        rrb.Vertical(
            rrb.Spatial2DView(origin="world/camera/rgb", name="Supporting RGB and masks"),
            rrb.Spatial2DView(origin="world/camera/depth", name="Supporting depth (m)"),
            rrb.TextDocumentView(origin="memory/visit", name="Visit, changes, and freshness"),
            row_shares=[1.,1.,1.15]), column_shares=[2.3,1.])
    blueprint = rrb.Blueprint(layout, rrb.BlueprintPanel(expanded=False),
                             rrb.SelectionPanel(expanded=False),
                             rrb.TimePanel(expanded=True, timeline="visit"),
                             auto_layout=False, auto_views=False)
    recording = output / "memory.rrd"
    rr.init("SCOPE persistent memory snapshots")
    rr.save(recording, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/axes", rr.TransformAxes3D(axis_length=.5), static=True)
    colors = {"VISIBLE":[68,207,180], "NOT_CURRENTLY_OBSERVED":[240,181,75],
              "POSSIBLY_MISSING":[240,102,113], "IDENTITY_UNCERTAIN":[180,140,236],
              "POSSIBLY_MOVED":[240,181,75]}
    for index, episode in enumerate(episodes):
        eid = episode["episode_id"]
        directory = store.verify_assets(eid)
        run = load_semantic_run(directory)
        T = np.asarray(episode["alignment"]["T_global_episode"])
        rr.set_time("visit", sequence=index+1)
        rr.set_time("elapsed", duration=episode["end_s"]-episodes[0]["end_s"])
        # Recompute these layers for each visit; earlier events stay on earlier ticks.
        for path in ("world/history", "world/changes", "world/negative_checks", "world/map"):
            rr.log(path, rr.Clear(recursive=True))
        if episode["map_reference"]:
            mapping = load_map(directory)
            points, rgb, _ = mapping.cloud()
            rr.log("world/map", rr.Points3D(camera_to_world(points,T), colors=rgb, radii=.009))
        beliefs = store.beliefs(eid)
        # Prefer a frame supporting the backpack; all frame indices remain in evidence.
        selected = len(run.frames)-1
        bags = [b for b in beliefs.values() if b["label"]=="backpack" and
                b["last_positive"]["episode_id"]==eid]
        if bags:
            selected = bags[0]["last_positive"]["asset_index"]
        frame = run.frames[selected]
        pose = T @ frame.T_world_camera
        rr.log("world/camera", rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]))
        K = frame.intrinsics
        rr.log("world/camera", rr.Pinhole(width=K.width, height=K.height,
            focal_length=[K.fx,K.fy], principal_point=[K.cx,K.cy],
            camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=.25))
        rgb = frame.rgb.astype(float).copy()
        for d in run.detections[selected]:
            rgb[d.mask] = .70*rgb[d.mask]+.30*np.array([68,207,180])
        rr.log("world/camera/rgb", rr.Image(rgb.astype(np.uint8)))
        rr.log("world/camera/depth", rr.DepthImage(np.nan_to_num(frame.depth_m), meter=1., depth_range=[0,5]))
        projected = run.projections[selected]
        if projected:
            rr.log("world/supporting_object_points", rr.Points3D(camera_to_world(
                np.concatenate([p.points_world for p in projected]), T), colors=[100,220,235], radii=.012))
        else:
            rr.log("world/supporting_object_points", rr.Clear(recursive=True))
        events = [e for e in store.history() if e["episode_id"]==eid]
        for n, (gid, belief) in enumerate(beliefs.items()):
            geometry = belief["geometry"]
            low, high = np.array(geometry["bounds_low_m"]), np.array(geometry["bounds_high_m"])
            path = f"world/current/{gid}"
            color = colors[belief["status"]]
            rr.log(path, rr.Boxes3D(centers=[(low+high)/2], sizes=[high-low], colors=color,
                                   fill_mode=rr.components.FillMode.MajorWireframe, radii=.014))
            position = np.array(geometry["center_m"])
            position[2] = high[2]+.17+.12*n
            freshness = episode["end_s"]-belief["last_positive"]["time_s"]
            label = f"{gid.removeprefix('global/')} [{belief['location_confidence']:.0%}]"
            if belief["status"]!="VISIBLE":
                label += f" {belief['status']}"
            rr.log(path+"/label", rr.Points3D([position], labels=[label], colors=color,
                                             show_labels=True, radii=.02))
            rr.log(path+"/evidence", rr.TextDocument(json.dumps({
                "belief": belief, "age_at_visit_end_s": freshness,
                "events": [e for e in events if e["global_id"]==gid]}, indent=2),
                media_type="application/json"))
            old_sightings = [e for e in store.history(gid) if e["kind"] in ("OBSERVED","REOBSERVED")
                            and e["time_s"]<episode["start_s"]]
            for old in old_sightings:
                old_belief = store.beliefs(old["episode_id"])[gid]
                g = old_belief["geometry"]
                old_low, old_high = np.array(g["bounds_low_m"]), np.array(g["bounds_high_m"])
                if np.linalg.norm(np.array(g["center_m"])-geometry["center_m"])<.20:
                    continue
                old_path = f"world/history/{gid}/{old['episode_id']}"
                rr.log(old_path, rr.Boxes3D(centers=[(old_low+old_high)/2], sizes=[old_high-old_low],
                       colors=[173,188,213,200], radii=.010,
                       show_labels=False,
                       fill_mode=rr.components.FillMode.MajorWireframe))
                rr.log(old_path+"/label", rr.Points3D(
                    [[g["center_m"][0],g["center_m"][1],old_high[2]+.12]],
                    labels=[f"HISTORY {gid.removeprefix('global/')} / {old['episode_id']}"],
                    colors=[173,188,213], show_labels=True, radii=.018))
        for event in events:
            if event["kind"] in ("MOVED","POSSIBLY_MOVED"):
                path = f"world/changes/{event['global_id']}/{event['seq']}"
                rr.log(path, rr.Arrows3D(origins=[event["from_m"]],
                    vectors=[np.array(event["to_m"])-event["from_m"]], colors=[245,202,90], radii=.02))
                midpoint = (np.array(event["from_m"])+event["to_m"])/2
                rr.log(path+"/label", rr.Points3D([midpoint], labels=[f"{event['kind']} {event['distance_m']:.2f} m"],
                                                 colors=[245,202,90], show_labels=True))
            if event["kind"] == "NOT_VISIBLE_FROM_VIEW":
                low, high = np.array(event["expected_bounds_low_m"]), np.array(event["expected_bounds_high_m"])
                origin = np.asarray(event["viewpoint_global"])[:3,3]
                rr.log(f"world/negative_checks/{event['seq']}", rr.LineStrips3D(
                    [[origin,(low+high)/2]], colors=[240,102,113], radii=.009))
        summary = [f"### Visit {index+1}: {eid}", f"Episode-end belief. Supporting frame: {frame.frame_id}", "",
                   "Green: positive · gray: history · amber: stale · red: missing · purple: uncertain", ""]
        for belief in beliefs.values():
            age = episode["end_s"]-belief["last_positive"]["time_s"]
            summary.append(f"- **{belief['global_id'].removeprefix('global/')}**: {belief['status']}, age {age:.1f} s")
        summary += [""]+[f"- **{e['kind']}**: {e['global_id']} ({e.get('confidence') or 0:.0%})"
                     for e in events if e["kind"] in ("MOVED","POSSIBLY_MOVED","MISSING_HYPOTHESIS","IDENTITY_UNCERTAIN")]
        rr.log("memory/visit", rr.TextDocument("\n".join(summary), media_type="text/markdown"))
    rr.disconnect()
    return recording
