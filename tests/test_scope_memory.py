"""Persistent evidence, cross-episode identity, and conservative change tests."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from scope.memory_store import MapAlignment, MemoryStore, episode_snapshot
from scope.semantic_pipeline import run_semantics
from scope.semantic_storage import load_semantic_run, save_semantic_run
from scope.storage import save_episode
from scope.synthetic import SyntheticRoom
from scope.truth import SyntheticTruthDetector


def saved_room(directory):
    room = SyntheticRoom(frames=5, width=96, height=72)
    frames = list(room)
    run = run_semantics(frames, SyntheticTruthDetector(room.boxes))
    save_episode(directory, frames, room.name)
    save_semantic_run(directory, run)
    return run


class MemoryStorageTests(unittest.TestCase):
    def test_replay_and_immutable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "episode-a"
            run = saved_room(directory)
            replay = load_semantic_run(directory)
            self.assertEqual(run.store.summaries(), replay.store.summaries())
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            snapshot = episode_snapshot("episode-a", directory, run,
                                        MapAlignment("room", "known_shared"))
            store.import_snapshot(snapshot)
            store.import_snapshot(snapshot)
            self.assertEqual(len(store.episodes()), 1)
            self.assertEqual(store.verify_assets("episode-a"), directory.resolve())
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("DELETE FROM episodes")
            changed = {**snapshot, "end_s": 99.}
            with self.assertRaises(ValueError):
                store.import_snapshot(changed)
            store.close()
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            self.assertEqual(store.episode("episode-a")["local_entities"],
                             snapshot["local_entities"])
            (directory / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                store.verify_assets("episode-a")
            store.close()

    def test_alignment_frame_and_rigidity_checks(self):
        import numpy as np
        invalid = np.eye(4)
        invalid[0, 0] = 2
        with self.assertRaises(ValueError):
            MapAlignment("room", "external_rigid", invalid).validate()
        with self.assertRaises(ValueError):
            MapAlignment("room", "unknown").validate()
        with self.assertRaises(ValueError):
            MapAlignment.from_dict({"world_frame":"room", "provenance":"external_rigid",
                                    "T_global_episode":np.eye(4).tolist()})


def memory_scenario(root, name):
    from scope.repeat_visits import scenario_visits, save_visit
    from scope.memory import process_episode
    store = MemoryStore(root / "memory.sqlite")
    visits = scenario_visits(name)
    for visit in visits:
        store.import_snapshot(save_visit(visit, root / visit.episode_id))
        process_episode(store, visit.episode_id)
    return store, visits


class GlobalIdentityTests(unittest.TestCase):
    def test_prior_alignment_uncertainty_also_blocks_identity(self):
        import numpy as np
        from scope.memory import process_episode
        from scope.repeat_visits import scenario_visits, save_visit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            visits = scenario_visits("unchanged")
            store = MemoryStore(root/"memory.sqlite")
            first = save_visit(visits[0],root/"a")
            first["alignment"]["translation_sigma_m"] = .25
            store.import_snapshot(first)
            process_episode(store,"episode_001")
            store.import_snapshot(save_visit(visits[1],root/"b"))
            links = process_episode(store,"episode_002")
            self.assertTrue(all(link["status"]=="UNRESOLVED" for link in links))
            self.assertEqual(len(store.beliefs()),4)
            T = np.eye(4)
            T[:3,3] = [100,100,100]
            alignment = MapAlignment("room","external_rigid",T,0.,.1)
            self.assertAlmostEqual(alignment.sigma_at_global([101,100,100]),.1)
            store.close()

    def test_supplied_rigid_alignment_restores_global_coordinates(self):
        from dataclasses import replace
        import numpy as np
        from scope.memory import process_episode
        from scope.repeat_visits import scenario_visits, save_visit
        from scope.memory_identity import aligned_entities
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            visits = scenario_visits("unchanged")
            store = MemoryStore(root/"memory.sqlite")
            store.import_snapshot(save_visit(visits[0],root/"a"))
            process_episode(store,"episode_001")
            T = np.eye(4)
            T[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
            T[:3,3] = [1.,.5,.2]
            frames = [replace(f,T_world_camera=np.linalg.inv(T)@f.T_world_camera)
                      for f in visits[1].run.frames]
            by_id = {f.frame_id:d for f,d in zip(frames,visits[1].run.detections)}
            class Masks:
                name = "synthetic-truth"
                def detect(self,frame): return by_id[frame.frame_id]
            run = run_semantics(frames,Masks())
            directory = root/"b"
            save_episode(directory,frames,"transformed synthetic")
            save_semantic_run(directory,run)
            snapshot = episode_snapshot("episode_002",directory,run,
                MapAlignment("synthetic-room","external_rigid",T),time_domain="unix_utc")
            store.import_snapshot(snapshot)
            links = process_episode(store,"episode_002")
            self.assertTrue(all(link["status"]=="LINKED" and link["score"]>.95 for link in links))
            for state in aligned_entities(snapshot):
                original = visits[1].run.store.entities[state["entity_id"]]
                self.assertLess(np.linalg.norm(np.array(state["center_m"])-original.center),.001)
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("UPDATE beliefs SET payload='{}'")
            store.close()

    def test_unchanged_and_moved_entities_retain_identity(self):
        from scope.memory import process_episode
        for scenario, moved in (("unchanged", None), ("moved_backpack", "backpack"),
                                ("moved_chair", "chair"), ("new_object", None)):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                store, visits = memory_scenario(Path(tmp), scenario)
                links = store.links("episode_002")
                self.assertEqual(sum(d["status"] == "LINKED" for d in links), 4)
                self.assertEqual(len(store.beliefs()), 5 if scenario == "new_object" else 4)
                events = [e for e in store.history() if e["kind"] == "MOVED"]
                self.assertEqual(len(events), 1 if moved else 0)
                if moved:
                    event = events[0]
                    self.assertIn(moved, event["global_id"])
                    self.assertNotEqual(event["from_m"], event["to_m"])
                    self.assertGreater(len(event["evidence_ids"]), 0)
                    old = store.beliefs("episode_001")[event["global_id"]]
                    self.assertEqual(old["geometry"]["center_m"], event["from_m"])
                    last = store.last_seen(event["global_id"])
                    self.assertEqual(last["last_positive"]["time_s"], visits[1].run.frames[-1].timestamp_s)
                count = len(store.history())
                process_episode(store, "episode_002")
                self.assertEqual(len(store.history()), count)
                store.close()


class NegativeEvidenceTests(unittest.TestCase):
    def test_missing_then_reappeared_restores_belief_without_erasing_events(self):
        from dataclasses import replace
        from scope.memory import process_episode
        from scope.repeat_visits import scenario_visits, save_visit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = memory_scenario(root,"removed")
            visit = scenario_visits("unchanged")[1]
            frames = [replace(f,timestamp_s=f.timestamp_s+86400,depth_timestamp_s=f.depth_timestamp_s+86400,
                              frame_id=f.frame_id.replace("episode_002","episode_003")) for f in visit.run.frames]
            visit.episode_id = "episode_003"
            visit.run = run_semantics(frames,SyntheticTruthDetector(visit.room.boxes))
            store.import_snapshot(save_visit(visit,root/visit.episode_id))
            process_episode(store,visit.episode_id)
            self.assertEqual(len(store.beliefs()),4)
            last = store.last_seen("global/backpack_0001")
            self.assertEqual(last["status"],"VISIBLE")
            self.assertEqual(last["last_positive"]["episode_id"],"episode_003")
            self.assertTrue(any(e["kind"]=="MISSING_HYPOTHESIS" for e in store.history("global/backpack_0001")))
            self.assertEqual(store.beliefs("episode_002")["global/backpack_0001"]["status"],"POSSIBLY_MISSING")
            store.close()
    def test_removed_unobserved_occluded_and_detector_miss(self):
        for scenario in ("removed", "unobserved", "occluded", "missing_detections"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                store, visits = memory_scenario(Path(tmp), scenario)
                gid = "global/backpack_0001"
                events = store.history(gid)
                missing = [e for e in events if e["kind"] == "MISSING_HYPOTHESIS"]
                self.assertEqual(len(missing), 1 if scenario == "removed" else 0)
                last = store.last_seen(gid)
                self.assertEqual(last["last_positive"]["episode_id"], "episode_001")
                self.assertEqual(last["last_positive"]["time_s"],
                                 visits[0].run.store.entities["backpack_01"].last_seen_s)
                checks = [e for e in events if "result" in e]
                self.assertEqual(len(checks), 8)
                if scenario == "removed":
                    self.assertEqual(last["status"], "POSSIBLY_MISSING")
                    self.assertGreaterEqual(last["later_negative_checks"], 2)
                    self.assertGreaterEqual(missing[0]["independent_negative_views"], 2)
                    self.assertTrue(all(e["coverage"]>.55 and e["valid_depth_fraction"]>.8
                                        for e in checks if e["result"]=="OBSERVED_ABSENT"))
                else:
                    self.assertEqual(last["status"], "NOT_CURRENTLY_OBSERVED")
                    self.assertEqual(last["later_negative_checks"], 0)
                store.close()


class MemoryEvaluationTests(unittest.TestCase):
    def test_metrics_check_changes_and_actual_evidence(self):
        from scope.memory_eval import evaluate_memory
        for scenario in ("moved_backpack","removed","ambiguous_chairs","alignment_10cm","new_object"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                store, visits = memory_scenario(Path(tmp),scenario)
                metrics = evaluate_memory(store,visits)
                self.assertEqual(metrics["false_identity_merges"],0)
                self.assertEqual(metrics["false_missing_claims"],0)
                self.assertEqual(metrics["last_seen_timestamp_correctness"],1.)
                self.assertEqual(metrics["evidence_retrieval_correctness"],1.)
                if scenario=="moved_backpack":
                    self.assertEqual(metrics["movement_recall"],1.)
                if scenario=="removed":
                    self.assertEqual(metrics["missing_hypothesis_recall"],1.)
                if scenario=="ambiguous_chairs":
                    self.assertEqual(metrics["unresolved_rate"],.5)
                    self.assertGreater(metrics["current_belief_position_error_mean_m"],.3)
                    self.assertEqual(metrics["current_updated_entity_position_error_count"],2)
                if scenario=="alignment_10cm":
                    self.assertGreater(metrics["current_updated_entity_position_error_mean_m"],.08)
                if scenario=="new_object":
                    self.assertEqual(metrics["current_updated_entity_position_error_count"],5)
                store.close()

    def test_bad_depth_and_unverified_detector_cannot_claim_absence(self):
        from dataclasses import replace
        import numpy as np
        from scope.memory_identity import aligned_entities
        from scope.memory_visibility import visibility_check
        from scope.repeat_visits import scenario_visits, save_visit
        with tempfile.TemporaryDirectory() as tmp:
            visits = scenario_visits("removed")
            snapshot = save_visit(visits[0], Path(tmp)/"a")
            bag = next(e for e in aligned_entities(snapshot) if e["label"]=="backpack")
            frame = visits[1].run.frames[1]
            invalid = replace(frame, depth_m=np.zeros_like(frame.depth_m))
            check = visibility_check(bag, invalid, np.eye(4), 1., 0.)
            self.assertEqual(check["result"], "LOW_DEPTH_QUALITY")
            check = visibility_check(bag, frame, np.eye(4), 0., 0.)
            self.assertEqual(check["result"], "DETECTOR_CAPABILITY_UNVERIFIED")

    def test_ambiguity_and_alignment_errors_abstain(self):
        for scenario in ("ambiguous_chairs", "alignment_25cm", "alignment_1m", "alignment_1m_undeclared"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                store, _ = memory_scenario(Path(tmp), scenario)
                unresolved = [d for d in store.links("episode_002") if d["status"] == "UNRESOLVED"]
                self.assertGreaterEqual(len(unresolved), 2)
                self.assertFalse(any(e["kind"] == "MOVED" for e in store.history()))
                self.assertTrue(all(d["candidates"] for d in unresolved))
                store.close()


if __name__ == "__main__":
    unittest.main()
