"""Rerun developer view: measured map, entities, people, rays and module ages."""

from pathlib import Path

import numpy as np

from .humans import BONES


class HumanRecorder:
    def __init__(self,output:Path,cameras):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr,self.started = rr,False
        self.output,self.cameras = output,tuple(cameras)
        self.blueprint = rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin="world",name="Map · entities · people · pointing",
                contents=["world/map/**","world/entities/**","world/humans/**","world/pointing/**",
                          "world/frustums/**","world/axes"],line_grid=False,background=[17,25,35],
                eye_controls=rrb.EyeControls3D(position=[4.,-6.,3.8],look_target=[0.,-.35,.85],eye_up=[0,0,1])),
            rrb.Vertical(*[rrb.Spatial2DView(origin=f"world/cameras/{c}/rgb",
                name=f"{c} RGB · joint quality") for c in cameras],
                rrb.Spatial2DView(origin=f"world/cameras/{cameras[0]}/depth",name="Aligned depth (m)")),
            rrb.Vertical(rrb.TextDocumentView(origin="dashboard/telemetry",name="Module health and age"),
                rrb.TextDocumentView(origin="dashboard/candidates",name="Pointing scores"),
                rrb.TextDocumentView(origin="dashboard/memory",name="M3 memory"),row_shares=[2.,1.2,.6]),
            column_shares=[2.2,1.,1.8]),rrb.BlueprintPanel(expanded=False),
            rrb.SelectionPanel(expanded=False),rrb.TimePanel(expanded=True,timeline="frame"),
            auto_layout=False,auto_views=False)

    def start(self):
        if not self.started:
            self.rr.init("SCOPE human perception")
            self.rr.save(self.output/"humans.rrd",default_blueprint=self.blueprint)
            self.rr.send_blueprint(self.blueprint,make_active=True,make_default=True)
            self.rr.log("world",self.rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
            self.rr.log("world/axes",self.rr.TransformAxes3D(axis_length=.5),static=True)
            self.started = True

    def log(self,pipeline,snapshot,index):
        return pipeline.runtime.run("rerun",snapshot["timestamp_s"],
            lambda:self._log(pipeline,snapshot,index),source="Rerun recording")

    def _log(self,pipeline,snapshot,index):
        rr,rt = self.rr,pipeline.runtime
        self.start()
        now = snapshot["timestamp_s"]
        rr.set_time("frame",sequence=index)
        rr.set_time("source_time",duration=now)
        for sample in snapshot["samples"]:
            f,c = sample.frame,sample.camera_id
            path = f"world/cameras/{c}"
            rr.log(path,rr.Transform3D(translation=f.T_world_camera[:3,3],mat3x3=f.T_world_camera[:3,:3]))
            K = f.intrinsics
            rr.log(path,rr.Pinhole(width=K.width,height=K.height,focal_length=[K.fx,K.fy],
                principal_point=[K.cx,K.cy],camera_xyz=rr.ViewCoordinates.RDF,image_plane_distance=.25))
            corners = np.array([[0,0],[K.width,0],[K.width,K.height],[0,K.height],[0,0]])
            local = np.column_stack(((corners[:,0]-K.cx)/K.fx,(corners[:,1]-K.cy)/K.fy,np.ones(5)))*.35
            world = local@f.T_world_camera[:3,:3].T+f.T_world_camera[:3,3]
            origin = f.T_world_camera[:3,3]
            rr.log(f"world/frustums/{c}",rr.LineStrips3D([world]+[[origin,p] for p in world[:4]],
                colors=[100,170,235],radii=.006))
            rr.log(f"world/frustums/{c}/pose",rr.Points3D([origin],labels=[c],colors=[100,170,235],
                radii=.025,show_labels=True))
            rr.log(path+"/rgb",rr.Image(f.rgb))
            rr.log(path+"/depth",rr.DepthImage(np.nan_to_num(f.depth_m,nan=0.),meter=1.,depth_range=[0,5]))
            rr.log(path+"/rgb/pose",rr.Clear(recursive=True))
            # Show oracle or actual model detections, never synthetic truth on
            # estimated runs. Runtime retains camera observations with their age.
            observations = rt.state(f"pose/{c}").last_output or []
            if not rt.state("human_pose").config.enabled or not rt.state(f"pose/{c}").config.enabled:
                observations = []
            for n,o in enumerate(observations):
                track = next((t.person_id for t in snapshot["tracks"] if o.observation_id in t.pose.observation_ids),"unresolved")
                keys = o.keypoints
                strips = [[keys[a].uv,keys[b].uv] for a,b in BONES if a in keys and b in keys
                          and min(keys[a].confidence,keys[b].confidence)>=.35]
                if strips:
                    rr.log(path+f"/rgb/pose/{n}/bones",rr.LineStrips2D(strips,colors=[105,240,178],radii=1.5))
                valid = [k for k in keys.values() if k.confidence>=.35]
                if valid:
                    rr.log(path+f"/rgb/pose/{n}/joints",rr.Points2D([k.uv for k in valid],
                        colors=[[255,int(100+155*k.confidence),80] for k in valid],radii=3.,
                        labels=[f"{k.name} {k.confidence:.2f}" for k in valid],show_labels=False))
                x1,y1,x2,y2 = o.box_xyxy
                age = max(0.,now-o.timestamp_s)
                rr.log(path+f"/rgb/pose/{n}/box",rr.Boxes2D(mins=[[x1,y1]],sizes=[[x2-x1,y2-y1]],
                    labels=[f"{track} · {age:.2f}s old"],colors=[105,240,178],show_labels=True))
        for c in self.cameras:
            if not rt.state(f"camera/{c}").config.enabled or c not in {s.camera_id for s in snapshot["samples"]}:
                rr.log(f"world/cameras/{c}",rr.Clear(recursive=True))
                rr.log(f"world/frustums/{c}",rr.Clear(recursive=True))
        def map_visualization():
            indices = np.argwhere(pipeline.mapping.states()==2)
            positions = np.asarray(pipeline.mapping.config.origin)+(indices+.5)*pipeline.mapping.config.voxel_m
            rr.log("world/map",rr.Points3D(positions,colors=[128,147,163],radii=.018))
        if rt.state("map_visualization").config.enabled:
            rt.run("map_visualization",now,map_visualization)
        else:
            rr.log("world/map",rr.Clear(recursive=True))
        for e in pipeline.entities.entities.values():
            rr.log(f"world/entities/{e.entity_id}",rr.Boxes3D(centers=[e.center],sizes=[e.high-e.low],
                colors=[239,181,79],radii=.01,fill_mode=rr.components.FillMode.MajorWireframe))
            center = e.center.copy()
            center[2] = e.high[2]+.10
            rr.log(f"world/entities/{e.entity_id}/label",rr.Points3D([center],labels=[e.entity_id],
                colors=[239,181,79],radii=.018,show_labels=True))
        rr.log("world/humans",rr.Clear(recursive=True))
        if rt.state("human_pose").config.enabled:
            for t in snapshot["tracks"]:
                joints = t.pose.joints
                strips = [[joints[a].position,joints[b].position] for a,b in BONES
                    if joints[a].position is not None and joints[b].position is not None]
                color = [105,240,178] if t.state=="VISIBLE" else [170,170,170]
                if strips:
                    rr.log(f"world/humans/{t.person_id}/bones",rr.LineStrips3D(strips,colors=color,radii=.014))
                valid = [j for j in joints.values() if j.position is not None]
                if valid:
                    rr.log(f"world/humans/{t.person_id}/uncertainty",rr.Points3D([j.position for j in valid],
                        colors=[*color,100],radii=[max(.018,np.sqrt(np.trace(j.covariance))) for j in valid]))
                    position = max((j.position for j in valid),key=lambda p:p[2])+[0,0,.15]
                    rr.log(f"world/humans/{t.person_id}/label",rr.Points3D([position],labels=[f"{t.person_id} {t.state}"],
                        colors=color,radii=.02,show_labels=True))
        rr.log("world/pointing",rr.Clear(recursive=True))
        lines = ["**Geometric pointing scores**", "Sampled hit fractions; uncalibrated."]
        for h in snapshot["pointing"]:
            lines.extend([f"\n**{h.person_track_id} · {h.state}**",h.reason])
            if h.state!="ABSTAIN":
                path = f"world/pointing/{h.person_track_id}"
                rr.log(path+"/central",rr.Arrows3D(origins=[h.origin],vectors=[h.direction*3.],
                    colors=[255,230,110],radii=.014))
                rr.log(path+"/samples",rr.LineStrips3D(
                    [[o,o+d*3.] for o,d in zip(h.origins[::8],h.directions[::8])],
                    colors=[255,200,80,75],radii=.003))
                lines.append(f"RMS angular spread: {np.rad2deg(h.angular_uncertainty_rad):.1f}°")
            result = snapshot["candidates"].get(h.person_track_id,{})
            for entity,score in result.get("scores",{}).items():
                lines.append(f"- {entity}: **{score:.3f}**")
            points = result.get("surface_points_m",[])
            if points:
                rr.log(f"world/pointing/{h.person_track_id}/surface_hits",rr.Points3D(points,
                    colors=[255,235,120],radii=.012))
        rr.log("dashboard/candidates",rr.TextDocument("\n".join(lines),media_type="text/markdown"))
        telemetry = rt.snapshot(now)
        rows = ["**Runtime · measured host rates**", "Joint colors show quality. Select a joint for its score.",
                "| Module | State | Hz | Age s | p95 ms | Drop |",
                "|---|---|---:|---:|---:|---:|"]
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        labels = {"semantic_detection":"semantics","human_pose":"humans",
            "world_entities":"entities","robot_visualization":"robot view",
            "map_visualization":"map view","visualization":"viewer",
            "memory_persist":"memory save","entity_projection":"projection",
            "acquire_input":"input","hand_pose":"hands"}
        for name,state in telemetry.items():
            rows.append(f"| {labels.get(name,name)} | {state['status']} | {fmt(state['host_update_hz'])} | "
                f"{fmt(state['data_age_s'])} | {fmt(state['latency']['p95_ms'])} | {state['dropped_inputs']} |")
        for name,state in telemetry.items():
            if state["status"] in ("DEGRADED","STALE","FAILED","UNAVAILABLE"):
                rows.append(f"\n**{name}:** {state['reason']}")
        rr.log("dashboard/telemetry",rr.TextDocument("\n".join(rows),media_type="text/markdown"))
        rr.log("dashboard/memory",rr.TextDocument(
            "**Persistent memory**\n"+"\n".join(f"- {k}: {v}" for k,v in pipeline.memory_status.items()),
            media_type="text/markdown"))

    def close(self):
        if self.started:
            self.rr.disconnect()
