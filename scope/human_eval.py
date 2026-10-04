"""Truth enters evaluation only, after pose association and target scoring."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .human_pipeline import HumanPipeline
from .human_synthetic import HumanScene, SCENARIOS
from .runtime import ModuleConfig


def mean(values):
    return float(np.mean(values)) if values else None


def evaluate_scene(scenario="complete",cameras=("front","left"),frames=10,width=128,distance_m=None):
    scene = HumanScene(frames,width,cameras,scenario,distance_m)
    pipeline = HumanPipeline(cameras,configs={"memory":ModuleConfig(enabled=False),
        "visualization":ModuleConfig(enabled=False),"rerun":ModuleConfig(enabled=False)})
    errors3d,errors2d,angles,rankings,hits,spreads,distances = [],[],[],[],[],[],[]
    seen,visible,missing,available,total_joints,abstentions,hypotheses = 0,0,0,0,0,0,0
    associations,track_ids,end_to_end = [],set(),[]
    expected_people,emitted_hypotheses = 0,0
    start = time.perf_counter()
    for i in range(frames):
        tick_start = time.perf_counter()
        samples = pipeline.runtime.run("acquire_input",i/10.,lambda:scene.tick(i),source=scene.name)
        snapshot = pipeline.step(samples,i/10.)
        expected_people += len(samples[0].truth) if samples else 0
        emitted_hypotheses += sum(h.state!="ABSTAIN" for h in snapshot["pointing"])
        end_to_end.append((time.perf_counter()-tick_start)*1000)
        for sample in samples:
            for truth,o in zip(sample.truth,sample.observations):
                visible += sum(truth["visibility"].values())
                for name,k in o.keypoints.items():
                    if truth["visibility"].get(name):
                        seen += 1
                        errors2d.append(float(np.linalg.norm(k.uv-truth["keypoints_2d"][name])))
        truth = samples[0].truth if samples else []
        associations.append(len(snapshot["unresolved_views"])+len(snapshot["unresolved_tracks"]))
        for track in snapshot["tracks"]:
            track_ids.add(track.person_id)
            # Evaluation associates a track to nearest truth shoulder, never
            # feeds this identity back into the inference/tracker.
            shoulder = track.pose.joints["right_shoulder"].position
            if shoulder is None or not truth or track.state!="VISIBLE":
                continue
            expected = min(truth,key=lambda t:np.linalg.norm(shoulder-t["keypoints_world"]["right_shoulder"]))
            for name,position in expected["keypoints_world"].items():
                total_joints += 1
                joint = track.pose.joints[name]
                if joint.position is None:
                    missing += 1
                else:
                    available += 1
                    errors3d.append(float(np.linalg.norm(joint.position-position)))
            h = next((h for h in snapshot["pointing"] if h.person_track_id==track.person_id),None)
            if h is None:
                continue
            hypotheses += 1
            if h.state=="ABSTAIN":
                abstentions += 1
                continue
            spreads.append(float(np.rad2deg(h.angular_uncertainty_rad)))
            target = expected["target_identity"]
            if target is None:
                continue
            angles.append(float(np.rad2deg(np.arccos(np.clip(h.direction@expected["ray_direction"],-1,1)))))
            distance = np.linalg.norm(expected["target"]-h.origin)
            distances.append(float(distance))
            scores = snapshot["candidates"][track.person_id]["scores"]
            # Semantic IDs are evaluated by their geometric center and class.
            candidates = [e for e in pipeline.entities.entities.values() if e.label=="chair"]
            correct = min(candidates,key=lambda e:np.linalg.norm(e.center-expected["target"])) if candidates else None
            if correct:
                ordered = sorted(scores,key=lambda k:-scores[k])
                rankings.append(ordered.index(correct.entity_id)+1 if scores[correct.entity_id]>0 else 999)
                hits.append(scores[correct.entity_id])
    wall = time.perf_counter()-start
    from .runtime import distribution
    result = {"scenario":scenario,"cameras":list(cameras),"frames":frames,"width":width,
        "pose_kind":"oracle visible 2D joints + simulated depth",
        "visible_joint_detection_rate":seen/visible if visible else None,
        "joint_error_2d_mean_px":mean(errors2d),"joint_error_3d_mean_m":mean(errors3d),
        "missing_joint_rate":missing/total_joints if total_joints else None,
        "angular_error_mean_deg":mean(angles),"angular_spread_mean_deg":mean(spreads),
        "target_hit_fraction_mean":mean(hits),
        "target_top1_rate":mean([r==1 for r in rankings]),
        "target_top3_recall":mean([r<=3 for r in rankings]),
        "abstention_rate":abstentions/hypotheses if hypotheses else None,
        "hypotheses":hypotheses,"track_ids_created":len(track_ids),
        "unresolved_associations":sum(associations),"target_distance_mean_m":mean(distances),
        "requested_target_distance_m":distance_m,
        "no_hypothesis_rate":1.-emitted_hypotheses/expected_people if expected_people else None,
        "wall_replay_fps":frames/wall,"host_perception_age":distribution(end_to_end),
        "telemetry":pipeline.runtime.snapshot((frames-1)/10.),
        "score_kind":"sampled geometric hit fraction; uncalibrated"}
    pipeline.close()
    return result


def benchmark(output,frames=10):
    output.mkdir(parents=True,exist_ok=True)
    cases = {}
    for scenario in SCENARIOS:
        for name,cameras in (("front",("front",)),("left",("left",)),("fused",("front","left"))):
            key = f"{scenario}/{name}"
            cases[key] = evaluate_scene(scenario,cameras,frames)
            print(f"{key}: angle={cases[key]['angular_error_mean_deg']} top1={cases[key]['target_top1_rate']} "
                  f"abstention={cases[key]['abstention_rate']}",flush=True)
    for distance in (.8,1.2,1.8):
        cases[f"distance/{distance}"] = evaluate_scene(frames=frames,distance_m=distance)
    for count in (1,2,3):
        cases[f"camera_count/{count}"] = evaluate_scene(frames=frames,
            cameras=("front","left","right")[:count])
    comparison = {}
    for scenario in SCENARIOS:
        single = [cases[f"{scenario}/{c}"] for c in ("front","left")]
        fused = cases[f"{scenario}/fused"]
        best = max(single,key=lambda r:r["target_top1_rate"] if r["target_top1_rate"] is not None else -1)
        comparison[scenario] = {"best_single_camera":best["cameras"][0],
            "best_single_top1":best["target_top1_rate"],"fused_top1":fused["target_top1_rate"],
            "best_single_3d_error_m":min((r["joint_error_3d_mean_m"] for r in single
                if r["joint_error_3d_mean_m"] is not None),default=None),
            "fused_3d_error_m":fused["joint_error_3d_mean_m"],
            "best_single_spread_deg":best["angular_spread_mean_deg"],
            "fused_spread_deg":fused["angular_spread_mean_deg"]}
    (output/"multi_camera_comparison.json").write_text(json.dumps(comparison,indent=2,allow_nan=False)+"\n")
    (output/"human_benchmark.json").write_text(json.dumps(cases,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = list(SCENARIOS)
    fig,axes = plt.subplots(3,1,figsize=(14,11),sharex=True)
    for mode in ("front","left","fused"):
        for ax,metric in zip(axes,("target_top1_rate","angular_spread_mean_deg","abstention_rate")):
            ax.plot(names,[cases[f"{n}/{mode}"][metric] if cases[f"{n}/{mode}"][metric] is not None
                          else np.nan for n in names],marker="o",label=mode)
            ax.set_ylabel(metric.replace("_"," "))
            ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].tick_params(axis="x",labelrotation=55)
    fig.suptitle("SCOPE virtual-camera geometry · oracle joints · uncalibrated scores")
    fig.tight_layout()
    fig.savefig(output/"human_benchmark.png",dpi=150)
    plt.close(fig)
    return cases
