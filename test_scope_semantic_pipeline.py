"""Independent semantic stages, evidence output, metrics, and optional real model."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scope.detector import clean_instance_mask
from scope.object_eval import (evaluate_synthetic, robustness_benchmark,
                               save_object_metrics, save_robustness)
from scope.relations import entity_relations
from scope.semantic_pipeline import run_semantics
from scope.semantic_storage import load_entity_evidence, save_semantic_run
from scope.synthetic import SyntheticRoom
from scope.truth import SyntheticTruthDetector


class SemanticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = SyntheticRoom(frames=5, width=128, height=96)
        cls.frames = list(cls.room)
        cls.detector = SyntheticTruthDetector(cls.room.boxes)

    def test_stages_can_run_independently(self):
        detected = run_semantics(self.frames, self.detector, stage="detect")
        self.assertGreater(sum(map(len, detected.detections)), 0)
        self.assertEqual(sum(map(len, detected.projections)), 0)
        self.assertIsNone(detected.store)
        projected = run_semantics(self.frames, self.detector, stage="project")
        self.assertEqual(sum(map(len, projected.detections)),
                         sum(map(len, projected.projections)))
        self.assertIsNone(projected.store)
        entities = run_semantics(self.frames, self.detector, stage="entities")
        self.assertEqual(len(entities.store.entities), 4)

    def test_metrics_and_evidence_roundtrip(self):
        run = run_semantics(self.frames, self.detector, stage="entities")
        metrics = evaluate_synthetic(run, self.room)
        self.assertEqual(metrics["false_merges"], 0)
        self.assertEqual(metrics["false_splits"], 0)
        self.assertEqual(metrics["id_switches"], 0)
        self.assertGreater(metrics["association_pair_accuracy"], .99)
        self.assertLess(metrics["entity_localization_error_mean_m"], .10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_semantic_run(path, run)
            save_object_metrics(path, metrics)
            (path / "relations.json").write_text(json.dumps(
                entity_relations(list(run.store.entities.values()))))
            evidence = load_entity_evidence(path, "chair_01")
            self.assertEqual(len(evidence["detections"]), 5)
            self.assertEqual(len(evidence["association_decisions"]), 5)
            self.assertTrue(any(item["target"] == "chair_02"
                                for item in evidence["relations"]))
            self.assertTrue((path / "semantic_quality.png").exists())
            self.assertEqual(json.loads((path / "semantic_metrics.json").read_text())
                             ["false_merges"], 0)
            with np.load(path / "semantic_masks.npz") as masks:
                self.assertEqual(len(masks.files), sum(map(len, run.detections)))
        relationships = entity_relations(list(run.store.entities.values()))
        self.assertTrue(any(item["source"] == "chair_01" and
                            item["target"] == "chair_02" and
                            "left_of_world_x" in item["relations"]
                            for item in relationships))

    def test_mask_cleanup_preserves_main_instance(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[4:20, 4:20] = True
        mask[30, 30] = True
        cleaned = clean_instance_mask(mask)
        self.assertTrue(cleaned[10, 10])
        self.assertFalse(cleaned[30, 30])


class RobustnessBenchmarkTests(unittest.TestCase):
    def test_controlled_failures_are_measured(self):
        cases = robustness_benchmark()
        self.assertEqual(len(cases), 7)
        self.assertEqual(cases["baseline"]["false_merges"], 0)
        self.assertEqual(cases["two_chairs_0p2m_gap"]["false_merges"], 0)
        self.assertEqual(cases["partial_chair_occlusion"]["false_splits"], 0)
        self.assertGreater(cases["depth_80pct_missing"]["projection_rejections"], 20)
        self.assertIsNone(cases["depth_80pct_missing"]["association_pair_accuracy"])
        self.assertGreater(cases["pose_shift_1m_one_frame"]["false_splits"], 0)
        self.assertGreater(cases["pose_shift_1m_one_frame"]["entity_localization_error_mean_m"],
                           cases["baseline"]["entity_localization_error_mean_m"] + .2)
        self.assertLess(cases["depth_noise_4cm_10pct_missing"]["entity_localization_error_mean_m"],
                        .10)
        with tempfile.TemporaryDirectory() as tmp:
            save_robustness(Path(tmp), cases)
            self.assertTrue((Path(tmp) / "semantic_robustness.png").exists())


@unittest.skipUnless(os.environ.get("SCOPE_REAL_DATASET"),
                     "Set SCOPE_REAL_DATASET to a local TUM RGB-D directory")
class RealDetectorIntegrationTests(unittest.TestCase):
    def test_mask_rcnn_to_world_entity(self):
        from scope.detector import TorchvisionMaskDetector
        from scope.tum import TumRgbd
        frames = list(TumRgbd(Path(os.environ["SCOPE_REAL_DATASET"]),
                              max_frames=2, frame_stride=10, width=320))
        run = run_semantics(frames, TorchvisionMaskDetector(
            include_classes={"chair", "keyboard", "cup", "tv"}), stage="entities")
        self.assertGreaterEqual(len(run.store.entities), 1)
        self.assertGreaterEqual(sum(map(len, run.projections)), 1)
        self.assertTrue(all(np.isfinite(entity.center).all()
                            for entity in run.store.entities.values()))


if __name__ == "__main__":
    unittest.main()
