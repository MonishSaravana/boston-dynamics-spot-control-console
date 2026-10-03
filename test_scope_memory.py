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


if __name__ == "__main__":
    unittest.main()
