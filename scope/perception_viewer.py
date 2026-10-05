"""Rerun receives detached snapshots on its own bounded worker."""

import numpy as np


class PerceptionRecorder:
    def __init__(self,output):
        self.output,self.started = output,False
        self.index = 0

    def log(self,s):
        import rerun as rr
        if not self.started:
            import rerun.blueprint as rrb
            blueprint = rrb.Blueprint(rrb.Horizontal(
                rrb.Spatial2DView(origin="scene/rgb",name="Requested objects · masks · tracks"),
                rrb.Spatial3DView(origin="world",name="Measured depth · entities · people",line_grid=False),
                rrb.Vertical(rrb.TextDocumentView(origin="query",name="Query evidence"),
                    rrb.TextDocumentView(origin="telemetry",name="Latency · backend · age")),column_shares=[1.5,1.3,1.4]),
                rrb.BlueprintPanel(expanded=False),rrb.SelectionPanel(expanded=False),auto_layout=False)
            rr.init("SCOPE object queries")
            rr.save(self.output/"perception.rrd",default_blueprint=blueprint)
            rr.send_blueprint(blueprint,make_active=True,make_default=True)
            rr.log("world",rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
            self.started = True
        self.index += 1
        rr.set_time("frame",sequence=self.index)
        frame = s["frame"]
        rr.log("scene/rgb",rr.Image(frame.rgb))
        rr.log("scene/rgb/targets",rr.Clear(recursive=True))
        for t in s["tracks"]:
            c = t["candidate"];x0,y0,x1,y1 = c["box_xyxy"]
            path = "scene/rgb/targets/"+t["track_id"]
            if t["mask"] is not None:
                rgba = np.zeros((*t["mask"].shape,4),np.uint8)
                rgba[t["mask"]] = [68,210,145,75]
                rr.log(path+"/mask",rr.Image(rgba))
            rr.log(path+"/box",rr.Boxes2D(mins=[[x0,y0]],sizes=[[x1-x0,y1-y0]],
                labels=[f'{t["track_id"]} · {t["state"]} · verified {t["verified_age_s"]:.2f}s ago'],colors=[68,210,145]))
        if "map" in s and s["telemetry"]["map_visualization"]["config"]["enabled"]:
            positions,colors,_ = s["map"]
            rr.log("world/map",rr.Points3D(positions,colors=colors,radii=.01))
        else:
            rr.log("world/map",rr.Clear(recursive=True))
        for e in s["entities"]:
            low,high = np.asarray(e["bounds_low_m"]),np.asarray(e["bounds_high_m"])
            rr.log("world/entities/"+e["entity_id"],rr.Boxes3D(centers=[(low+high)/2],sizes=[high-low],
                labels=[e["entity_id"]],colors=[235,183,83]))
        human = s.get("human_snapshot",{})
        from .humans import BONES
        rr.log("world/humans",rr.Clear(recursive=True))
        for t in human.get("tracks",[]):
            joints = t["joints"]
            lines = [[joints[a]["position_m"],joints[b]["position_m"]] for a,b in BONES
                     if joints[a]["position_m"] is not None and joints[b]["position_m"] is not None]
            rr.log("world/humans/"+t["person_id"],rr.LineStrips3D(lines,colors=[88,155,245],radii=.015))
        query_lines = []
        for q in s["queries"]:
            query_lines.extend([f'**{q["user_phrase"]} — {q["state"]}**',
                f'Expanded: {", ".join(q["expanded_queries"])}',q["reason"],
                f'Snapshot age: {q["age_s"]:.2f}s; detector scores are uncalibrated'])
            query_lines.extend(f'{c["detector_match"]}: {c["score"]:.3f}; mask {c["mask_available"]}; entity {c["entity_id"]}' for c in q["candidates"])
        rr.log("query",rr.TextDocument("\n\n".join(query_lines),media_type="text/markdown"))
        lines = ["Module | State | Hz | Age s | Last ms | p95 ms", "---|---|---:|---:|---:|---:"]
        def number(v):return "—" if v is None else f"{v:.2f}"
        for name,t in s["telemetry"].items():
            if t["successes"] or t["status"] in ("DISABLED","FAILED"):
                lines.append(f'{name} | {t["status"]} | {number(t["host_update_hz"])} | {number(t["data_age_s"])} | {number(t["current_latency_ms"])} | {number(t["latency"]["p95_ms"])}')
                if t["model"]:lines.append(f' | {t["device"]}: {t["model"]} | | | |')
        rss=s["resources"].get("rss_bytes")
        lines.extend(["",f'Core tick: {s["core_tick_ms"]:.2f} ms; cameras: {s["active_camera_count"]}',
            f'RSS: {number(rss/1e6 if rss is not None else None)} MB; CPU: {number(s["resources"].get("cpu_percent"))}%',
            "Accelerator utilization unavailable; memory records verified query events."])
        rr.log("telemetry",rr.TextDocument("\n".join(lines),media_type="text/markdown"))
