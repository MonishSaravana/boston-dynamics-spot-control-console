"""Synthetic cross-session identity, change, location, and retrieval metrics."""

import json
from pathlib import Path

import numpy as np

from .memory import process_episode
from .memory_store import MemoryStore
from .repeat_visits import SCENARIOS, Visit, save_visit, scenario_visits


def _fraction(numerator, denominator):
    return numerator/denominator if denominator else None


def evaluate_memory(store: MemoryStore, visits: list[Visit]) -> dict:
    # Truth is consumed only here, after all identity and change decisions.
    local_truth, expected_local = {}, {}
    visit_by_id = {v.episode_id:v for v in visits}
    for visit in visits:
        for local in visit.run.store.entities.values():
            ids = [visit.run.store.observations[oid].detection.truth_id for oid in local.observation_ids]
            identity = max(set(ids),key=ids.count)
            local_truth[visit.episode_id,local.entity_id] = identity
            expected_local[visit.episode_id,local.entity_id] = local
    links = store.links()
    global_truth = {}
    truth_globals = {}
    first_truth = {}
    for link in links:
        eid = link["episode_id"]
        truth = local_truth[eid,link["local_id"]]
        gid = link["global_id"]
        if gid:
            first_truth.setdefault(gid,truth)
            global_truth.setdefault(gid,set()).add(truth)
            truth_globals.setdefault(truth,set()).add(gid)
    # Correct accepted links versus identities actually observed in the next visit.
    second = visits[-1]
    second_links = store.links(second.episode_id)
    expected_existing = linked = correct = unresolved = 0
    score_errors = []
    for link in second_links:
        truth = local_truth[second.episode_id,link["local_id"]]
        expected_existing += truth in visits[0].truth
        unresolved += link["status"]=="UNRESOLVED"
        if link["status"]=="LINKED":
            linked += 1
            is_correct = first_truth[link["global_id"]]==truth
            correct += is_correct
            position = np.asarray(expected_local[second.episode_id,link["local_id"]].center)
            transform = second.alignment.T_global_episode
            position = position @ transform[:3,:3].T + transform[:3,3]
            score_errors.append({"score":link["score"],"correct_identity":bool(is_correct),
                "localization_error_m":float(np.linalg.norm(position-second.truth[truth]["center_m"])),
                "global_id":link["global_id"]})
    events = store.history()
    expected_moves = {truth for truth in visits[0].truth.keys() & second.truth.keys()
                      if np.linalg.norm(np.array(visits[0].truth[truth]["center_m"])-
                                        second.truth[truth]["center_m"])>.20}
    expected_new = set(second.truth)-set(visits[0].truth)
    expected_missing = set(visits[0].truth)-set(second.truth)
    moves = [e for e in events if e["episode_id"]==second.episode_id and e["kind"]=="MOVED"]
    news = [e for e in events if e["episode_id"]==second.episode_id and e["kind"]=="NEW_ENTITY"]
    missing = [e for e in events if e["episode_id"]==second.episode_id and e["kind"]=="MISSING_HYPOTHESIS"]
    correct_moves = {first_truth[e["global_id"]] for e in moves if first_truth[e["global_id"]] in expected_moves}
    correct_new = {first_truth[e["global_id"]] for e in news if first_truth[e["global_id"]] in expected_new}
    correct_missing = {first_truth[e["global_id"]] for e in missing if first_truth[e["global_id"]] in expected_missing}
    historical_errors, timestamp_correct, location_correct, evidence_correct = [],[],[],[]
    current_errors, updated_errors = [],[]
    for gid, belief in store.beliefs().items():
        truth = first_truth[gid]
        if truth in second.truth:
            error = float(np.linalg.norm(np.array(belief["geometry"]["center_m"])-
                                         second.truth[truth]["center_m"]))
            current_errors.append(error)
            if belief["last_positive"]["episode_id"]==second.episode_id:
                updated_errors.append(error)
        last = store.last_seen(gid)
        positive = last["last_positive"]
        visit = visit_by_id[positive["episode_id"]]
        local = expected_local[positive["episode_id"],positive["local_id"]]
        identity = local_truth[positive["episode_id"],positive["local_id"]]
        timestamp_correct.append(positive["time_s"]==local.last_seen_s)
        location_correct.append(np.linalg.norm(np.array(positive["position_m"])-
                                               visit.truth[identity]["center_m"])<.15)
        observed = visit.run.store.observations
        frame = visit.run.frames[positive["asset_index"]]
        evidence_correct.append(frame.frame_id==positive["frame_id"] and all(
            oid in observed for oid in positive["observation_ids"]))
    for event in events:
        if event["kind"] not in ("OBSERVED","REOBSERVED"):
            continue
        positive = event["positive"]
        identity = local_truth[event["episode_id"],positive["local_id"]]
        historical_errors.append(float(np.linalg.norm(np.array(event["position_m"])-
                                                      visit_by_id[event["episode_id"]].truth[identity]["center_m"])))
    switches = 0
    for truth in truth_globals:
        sequence = [link["global_id"] for visit in visits for link in store.links(visit.episode_id)
                    if link["global_id"] and local_truth[visit.episode_id,link["local_id"]]==truth]
        switches += sum(a!=b for a,b in zip(sequence,sequence[1:]))
    return {
        "global_id_precision":_fraction(correct,linked),
        "global_id_recall":_fraction(correct,expected_existing),
        "false_identity_merges":sum(len(ids)>1 for ids in global_truth.values()),
        "false_identity_splits":sum(max(0,len(ids)-1) for ids in truth_globals.values()),
        "identity_switches":switches,
        "unresolved_rate":_fraction(unresolved,len(second_links)),
        "accepted_links":linked,"expected_existing_local_entities":expected_existing,
        "movement_precision":_fraction(len(correct_moves),len(moves)),
        "movement_recall":_fraction(len(correct_moves),len(expected_moves)),
        "new_object_precision":_fraction(len(correct_new),len(news)),
        "new_object_recall":_fraction(len(correct_new),len(expected_new)),
        "missing_hypothesis_precision":_fraction(len(correct_missing),len(missing)),
        "missing_hypothesis_recall":_fraction(len(correct_missing),len(expected_missing)),
        "false_missing_claims":len(missing)-len(correct_missing),
        "moved_events":len(moves),"new_events":len(news),"missing_events":len(missing),
        "current_belief_position_error_mean_m":float(np.mean(current_errors)) if current_errors else None,
        "current_belief_position_error_count":len(current_errors),
        "current_updated_entity_position_error_mean_m":float(np.mean(updated_errors)) if updated_errors else None,
        "current_updated_entity_position_error_count":len(updated_errors),
        "historical_position_error_mean_m":float(np.mean(historical_errors)) if historical_errors else None,
        "last_seen_timestamp_correctness":float(np.mean(timestamp_correct)) if timestamp_correct else None,
        "last_seen_location_within_0_15m":float(np.mean(location_correct)) if location_correct else None,
        "evidence_retrieval_correctness":float(np.mean(evidence_correct)) if evidence_correct else None,
        "identity_score_errors":score_errors,
        "truth_kind":"deterministic synthetic boxes and exact masks; truth never enters identity scoring",
    }


def benchmark_memory(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new benchmark directory to preserve prior results")
    output.mkdir(parents=True,exist_ok=True)
    cases = {}
    for name in SCENARIOS:
        directory = output/name
        store = MemoryStore(directory/"memory.sqlite")
        try:
            visits = scenario_visits(name)
            for visit in visits:
                store.import_snapshot(save_visit(visit,directory/visit.episode_id))
                process_episode(store,visit.episode_id)
            cases[name] = evaluate_memory(store,visits)
            (directory/"memory_metrics.json").write_text(json.dumps(cases[name],indent=2)+"\n")
        finally:
            store.close()
    (output/"memory_metrics.json").write_text(json.dumps(cases,indent=2)+"\n")
    _plots(output,cases)
    return {name:{k:v for k,v in metrics.items() if k not in ("identity_score_errors","truth_kind")}
            for name,metrics in cases.items()}


def _plots(output,cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cases)
    y = np.arange(len(names))
    fig,axes = plt.subplots(1,2,figsize=(12,7),layout="constrained")
    axes[0].barh(y-.18,[cases[n]["global_id_recall"] or 0 for n in names],height=.35,label="Identity recall",color="#4aa8b5")
    axes[0].barh(y+.18,[cases[n]["unresolved_rate"] or 0 for n in names],height=.35,label="Unresolved",color="#e6a056")
    axes[0].set_yticks(y,names); axes[0].invert_yaxis(); axes[0].set_xlim(0,1); axes[0].legend()
    values = [cases[n]["current_updated_entity_position_error_mean_m"] for n in names]
    axes[1].barh(y,[v or 0 for v in values],color="#8d92d2")
    axes[1].set_yticks(y,names); axes[1].invert_yaxis(); axes[1].set_xlabel("Mean updated entity position error (m)")
    for index,value in enumerate(values):
        if value is None: axes[1].text(.002,index,"N/A: no positive updates",va="center",fontsize=8)
    fig.suptitle("Synthetic persistent-memory robustness; abstention is explicit")
    fig.savefig(output/"memory_robustness.png",dpi=150); plt.close(fig)
    alignment = ["unchanged","alignment_10cm","alignment_25cm","alignment_1m"]
    error = [0.,.1,.25,1.]
    fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
    ax.plot(error,[cases[n]["global_id_recall"] for n in alignment],"o-",label="Identity recall")
    ax.plot(error,[cases[n]["unresolved_rate"] for n in alignment],"o-",label="Unresolved rate")
    ax.set(xlabel="Applied and declared translation error / uncertainty (m)",ylabel="Fraction",ylim=(-.03,1.03))
    ax.legend(); ax.set_title("Alignment uncertainty changes linking and abstention")
    fig.savefig(output/"alignment_sensitivity.png",dpi=150); plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
    for name in ("unchanged","moved_backpack","moved_chair","depth_noise","alignment_10cm"):
        points = cases[name]["identity_score_errors"]
        ax.scatter([p["score"] for p in points],[p["localization_error_m"] for p in points],label=name)
    ax.set(xlabel="Accepted identity score (heuristic)",ylabel="Position error (m)")
    ax.legend(fontsize=8); ax.set_title("Localization error versus identity score; rejected links excluded")
    fig.savefig(output/"identity_score_error.png",dpi=150); plt.close(fig)
