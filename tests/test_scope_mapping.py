"""Milestone 1: geometric truth, failure handling, replay, and dataset input."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scope.data import Intrinsics, camera_to_world, project_depth
from scope.evaluate import surface_distance, synthetic_metrics
from scope.mapping import DenseTsdf, FREE, OCCUPIED, UNKNOWN, MapConfig, VoxelMap
from scope.storage import (load_episode, load_map, load_tsdf, save_episode, save_map,
                           save_tsdf)
from scope.synthetic import SyntheticRoom
from scope.tum import TumRgbd


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = SyntheticRoom(frames=6, width=96, height=72)
        cls.frames = list(cls.scene)

    def test_projection_and_transform_hit_analytic_scene(self):
        frame = self.frames[0]
        points, _, _ = project_depth(frame, stride=2)
        world = camera_to_world(points, frame.T_world_camera)
        self.assertLess(float(np.percentile(surface_distance(world, self.scene.boxes), 99)),
                        0.001)
        self.assertTrue(np.allclose(frame.T_world_camera[:3, :3].T @
                                    frame.T_world_camera[:3, :3], np.eye(3)))

    def test_unknown_behind_chair_is_not_free(self):
        mapping = VoxelMap(MapConfig.synthetic())
        self.assertTrue(mapping.integrate(self.frames[0]))
        state = mapping.states()
        to_index = lambda p: tuple(np.floor((np.array(p) - mapping.config.origin) /
                                            mapping.config.voxel_m).astype(int))
        self.assertEqual(state[to_index((1., .9, .8))], UNKNOWN)
        self.assertEqual(state[to_index((0., 0., .6))], FREE)
        self.assertEqual(state[to_index((1.2, .49, .8))], OCCUPIED)

    def test_reconstruction_metrics_and_pose_error(self):
        normal = VoxelMap(MapConfig.synthetic())
        corrupt = VoxelMap(MapConfig.synthetic())
        for i, frame in enumerate(self.frames):
            normal.integrate(frame)
            if i == 3:
                pose = frame.T_world_camera.copy()
                pose[:3, 3] += [1.0, 0, 0]
                frame = replace(frame, T_world_camera=pose)
            corrupt.integrate(frame)
        expected = synthetic_metrics(normal, self.scene.boxes)
        shifted = synthetic_metrics(corrupt, self.scene.boxes)
        self.assertGreater(expected["free_space_precision"], .95)
        self.assertGreater(expected["unknown_fraction"], .5)
        self.assertGreater(shifted["surface_error_mean_m"],
                           expected["surface_error_mean_m"] + .005)

    def test_bad_time_calibration_depth_and_transform(self):
        first = self.frames[0]
        mapping = VoxelMap(MapConfig.synthetic())
        self.assertTrue(mapping.integrate(first))
        late = replace(self.frames[1], depth_timestamp_s=self.frames[1].timestamp_s + .2)
        self.assertFalse(mapping.integrate(late))
        self.assertIn("timing", mapping.rejected[-1])
        bad_k = replace(self.frames[1], intrinsics=Intrinsics(
            first.intrinsics.width, first.intrinsics.height,
            first.intrinsics.fx + 50, first.intrinsics.fy,
            first.intrinsics.cx, first.intrinsics.cy))
        self.assertFalse(mapping.integrate(bad_k))
        self.assertIn("calibration", mapping.rejected[-1])
        missing = replace(self.frames[1], depth_m=np.zeros_like(first.depth_m))
        self.assertFalse(mapping.integrate(missing))
        self.assertIn("no valid depth", mapping.rejected[-1])
        pose = first.T_world_camera.copy()
        pose[:3, :3] *= 2
        self.assertFalse(mapping.integrate(replace(self.frames[1], T_world_camera=pose)))
        self.assertIn("not rigid", mapping.rejected[-1])
        self.assertEqual(mapping.revision, 1)

    def test_tsdf_surface_and_saved_replay(self):
        config = MapConfig.synthetic()
        mapping = VoxelMap(config)
        tsdf = DenseTsdf(config)
        for frame in self.frames:
            mapping.integrate(frame)
            tsdf.integrate(frame)
        mesh = tsdf.mesh()
        self.assertGreater(len(mesh.triangles), 100)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_episode(directory, self.frames, self.scene.name)
            save_map(directory, mapping)
            save_tsdf(directory, tsdf)
            replay = load_episode(directory)
            loaded = load_map(directory)
            loaded_tsdf = load_tsdf(directory, config)
            again = VoxelMap(config)
            for frame in replay:
                again.integrate(frame)
            self.assertTrue(np.array_equal(mapping.states(), loaded.states()))
            self.assertTrue(np.array_equal(mapping.hits, again.hits))
            self.assertTrue(np.array_equal(tsdf.weight, loaded_tsdf.weight))

    def test_tum_directory_with_known_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            Image.new("RGB", (640, 480), (20, 30, 40)).save(root / "rgb" / "a.png")
            Image.fromarray(np.full((480, 640), 5000, dtype=np.uint16)).save(
                root / "depth" / "b.png")
            (root / "rgb.txt").write_text("1.000 rgb/a.png\n")
            (root / "depth.txt").write_text("1.012 depth/b.png\n")
            (root / "groundtruth.txt").write_text("1.005 0 0 0 0 0 0 1\n")
            frames = list(TumRgbd(root, width=160))
            self.assertEqual(len(frames), 1)
            frame = frames[0]
            frame.validate()
            self.assertEqual(frame.rgb.shape, (120, 160, 3))
            self.assertAlmostEqual(float(frame.depth_m[20, 20]), 1.)
            self.assertTrue(np.allclose(frame.T_world_camera, np.eye(4)))


if __name__ == "__main__":
    unittest.main()
