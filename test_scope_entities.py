"""Episode-local association under viewpoint, occlusion, and semantic changes."""

from dataclasses import replace
import unittest

import numpy as np

from scope.entities import EntityStore
from scope.objects import project_object
from scope.synthetic import Box, SyntheticRoom
from scope.truth import SyntheticTruthDetector


class EntityFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = SyntheticRoom(frames=10)
        cls.frames = list(cls.room)
        cls.detector = SyntheticTruthDetector(cls.room.boxes)
        cls.projected = [[p for detection in cls.detector.detect(frame)
                          if (p := project_object(frame, detection)) is not None]
                         for frame in cls.frames]

    def test_repeated_views_stable_ids_and_evidence(self):
        store = EntityStore()
        for frame in self.projected:
            store.add_frame(frame)
        self.assertEqual(set(store.entities),
                         {"chair_01", "chair_02", "table_01", "backpack_01"})
        for entity in store.entities.values():
            truth_ids = {store.observations[oid].detection.truth_id
                         for oid in entity.observation_ids}
            self.assertEqual(len(truth_ids), 1)
            truth_id = next(iter(truth_ids))
            expected_views = {frame.frame_id for frame in self.frames
                              if any(d.truth_id == truth_id
                                     for d in self.detector.detect(frame))}
            self.assertEqual(len(entity.observation_ids), len(expected_views))
            self.assertEqual(entity.supporting_views, expected_views)
            self.assertEqual(entity.first_seen_s, 0.)
            self.assertEqual(entity.last_seen_s,
                             max(self.frames[int(frame_id.split("-")[-1])].timestamp_s
                                 for frame_id in expected_views))
            self.assertGreater(entity.confidence, .85)
        self.assertEqual(len(store.decisions), sum(map(len, self.projected)))
        self.assertTrue(all("candidates" in d for d in store.decisions))

    def test_nearby_same_class_chairs_stay_distinct(self):
        room = SyntheticRoom(frames=6)
        boxes = []
        for part in room.boxes:
            if part.name.startswith("table") or part.name == "backpack":
                continue
            if part.name.startswith("chair B"):
                shift = np.array([-1.22, 0., 0.])
                boxes.append(Box(part.name, tuple(np.array(part.low) + shift),
                                 tuple(np.array(part.high) + shift), part.color))
            else:
                boxes.append(part)
        room.boxes = tuple(boxes)
        detector = SyntheticTruthDetector(room.boxes)
        store = EntityStore()
        for frame in room:
            store.add_frame([p for detection in detector.detect(frame)
                             if (p := project_object(frame, detection)) is not None])
        self.assertEqual(set(store.entities), {"chair_01", "chair_02"})
        self.assertEqual({next(iter({store.observations[oid].detection.truth_id
                                     for oid in e.observation_ids}))
                          for e in store.entities.values()}, {"chair_a", "chair_b"})
        self.assertTrue(all(len(e.observation_ids) >= 4 for e in store.entities.values()))

    def test_partial_occlusion_and_reappearance(self):
        store = EntityStore()
        first = next(o for o in self.projected[0] if o.label == "chair")
        store.add_frame([first])
        later = next(o for o in self.projected[2]
                     if o.detection.truth_id == "chair_a")
        mask = later.detection.mask.copy()
        midpoint = (later.detection.box_xyxy[0] + later.detection.box_xyxy[2]) // 2
        mask[:, midpoint:] = False
        occluded = project_object(self.frames[2],
                                  replace(later.detection, mask=mask,
                                          observation_id="partial-chair-a"))
        self.assertIsNotNone(occluded)
        store.add_frame([occluded])
        for _ in range(3):
            store.add_frame([])
        reappeared = next(o for o in self.projected[9]
                          if o.detection.truth_id == "chair_a")
        store.add_frame([reappeared])
        self.assertEqual(set(store.entities), {"chair_01"})
        self.assertEqual(len(store.entities["chair_01"].observation_ids), 3)

    def test_class_disagreement_fuses_probabilities(self):
        store = EntityStore()
        first = next(o for o in self.projected[0]
                     if o.detection.truth_id == "chair_a")
        second = next(o for o in self.projected[1]
                      if o.detection.truth_id == "chair_a")
        conflicting = replace(second, detection=replace(
            second.detection, class_probabilities={"chair": .35, "stool": .65}))
        store.add_frame([first])
        store.add_frame([conflicting])
        self.assertEqual(set(store.entities), {"chair_01"})
        probabilities = store.entities["chair_01"].class_probabilities
        self.assertGreater(probabilities["chair"], probabilities["stool"])
        self.assertGreater(probabilities["stool"], 0)

    def test_pose_perturbation_does_not_silently_merge(self):
        store = EntityStore()
        first = next(o for o in self.projected[0]
                     if o.detection.truth_id == "chair_a")
        store.add_frame([first])
        frame = self.frames[1]
        detection = next(d for d in self.detector.detect(frame)
                         if d.truth_id == "chair_a")
        shifted = frame.T_world_camera.copy()
        shifted[:3, 3] += [1.0, 0., 0.]
        altered = project_object(replace(frame, T_world_camera=shifted), detection)
        self.assertIsNotNone(altered)
        store.add_frame([altered])
        self.assertEqual(len(store.entities), 2)
        self.assertNotEqual(store.observation_entity[first.observation_id],
                            store.observation_entity[altered.observation_id])

    def test_bounded_history_preserves_full_fusion_statistics(self):
        bounded,full=EntityStore(history_limit=3,entity_limit=8),EntityStore()
        for observations in self.projected:
            bounded.add_frame(observations);full.add_frame(observations)
        self.assertLessEqual(len(bounded.observations),3)
        self.assertLessEqual(len(bounded.decisions),3)
        for identity,e in bounded.entities.items():
            self.assertLessEqual(len(e.observation_ids),3)
            self.assertEqual(e.observation_count,full.entities[identity].observation_count)
            np.testing.assert_allclose(e.center,full.entities[identity].center)
            np.testing.assert_allclose(e.covariance,full.entities[identity].covariance)


if __name__ == "__main__":
    unittest.main()
