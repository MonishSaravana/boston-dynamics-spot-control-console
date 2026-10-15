"""Model choices are validated and applied without loading real weights."""
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scope import model_settings
from scope.console_backend import SpotBackend
from scope.workspace import Workspace
from scope_web import WebControl


class FakeDetector:
    def __init__(self, name):
        self.name = name
        self.size, self.threshold, self.low_threshold = 512, .4, .15
        self.metadata = {"model": name}


class FakeLibrary:
    def __init__(self):
        self.loaded = {}

    def detector(self, backend, device):
        return self.loaded.setdefault((backend, device), FakeDetector(backend))

    def segmenter(self, device):
        return self.loaded.setdefault(("sam2", device), object())


class ModelSettingsTests(unittest.TestCase):
    def test_every_default_and_parameter_names_a_real_choice(self):
        for option in model_settings.OPTIONS:
            values = {c["value"] for c in option["choices"]}
            self.assertIn(model_settings.DEFAULTS[option["key"]], values)
            for choice in option["choices"]:
                self.assertTrue(choice["label"] and choice["summary"])
        self.assertEqual(set(model_settings.THRESHOLDS), self._values("confidence"))
        self.assertEqual(set(model_settings.SIZES), self._values("detail"))
        self.assertEqual(set(model_settings.REFRESH_S), self._values("refresh"))

    def _values(self, key):
        option = next(o for o in model_settings.OPTIONS if o["key"] == key)
        return {c["value"] for c in option["choices"]}

    def test_validate_fills_defaults_and_rejects_unknown_values(self):
        values = model_settings.validate({"finder": "fast"})
        self.assertEqual(values, {**model_settings.DEFAULTS, "finder": "fast"})
        for bad in ({"finder": "magic"}, {"speed": "fast"}, ["fast"], None):
            with self.assertRaises(ValueError):
                model_settings.validate(bad)

    def test_fast_finder_is_configured_dino_without_mutating_loaded_weights(self):
        library = FakeLibrary()
        values = {**model_settings.DEFAULTS, "finder": "fast", "confidence": "loose",
                  "device": "cpu"}
        finder = model_settings.query_factory(values, library)()
        loaded = library.loaded[("grounding-dino", "cpu")]
        self.assertIsNot(finder, loaded)
        self.assertEqual((finder.size, finder.threshold), (512, .35))
        self.assertEqual(finder.metadata["acceptance_threshold"], .35)
        self.assertEqual(loaded.threshold, .4)
        self.assertNotIn(("owlv2", "cpu"), library.loaded)

    def test_checked_high_detail_uses_both_models_at_higher_resolution(self):
        library = FakeLibrary()
        values = {**model_settings.DEFAULTS, "detail": "high", "confidence": "strict",
                  "device": "cpu"}
        finder = model_settings.query_factory(values, library)()
        self.assertIsInstance(finder, model_settings.CheckedFinder)
        self.assertEqual((finder.primary.size, finder.primary.threshold), (800, .55))
        self.assertEqual((finder.verifier.size, finder.verifier.threshold), (768, .15))
        self.assertEqual(finder.metadata["verification"]["minimum_box_iou"], .3)

    def test_outline_choice_selects_grabcut_or_cached_sam(self):
        from scope.query_segmentation import GrabCutSegmenter
        library = FakeLibrary()
        self.assertIs(model_settings.segmenter_factory(model_settings.DEFAULTS, library),
                      GrabCutSegmenter)
        sam = model_settings.segmenter_factory(
            {**model_settings.DEFAULTS, "outline": "sam2", "device": "cpu"}, library)
        self.assertIs(sam(), sam())

    def test_library_loads_once_and_frees_other_devices(self):
        library = model_settings.ModelLibrary()
        build = Mock(side_effect=lambda: object())
        first = library._get(("detector", "owlv2", "mps"), build)
        self.assertIs(library._get(("detector", "owlv2", "mps"), build), first)
        self.assertEqual(build.call_count, 1)
        library.keep_only("cpu")
        self.assertEqual(library.loaded, {})

    def test_apply_swaps_models_and_refresh_without_touching_the_map(self):
        state = SimpleNamespace(durations_ms=deque([900.]), cpu_ms=deque([800.]))
        mapping = object()
        pipeline = SimpleNamespace(query_detector=object(), segmenter=object(),
            query_factory=None, segmenter_factory=None, requery_s=1.5,
            last_query_submit_s=10., runtime=SimpleNamespace(state=lambda name: state),
            core=SimpleNamespace(mapping=mapping))
        backend = SpotBackend.__new__(SpotBackend)
        backend.pipeline, backend.models = pipeline, FakeLibrary()
        values = {**model_settings.DEFAULTS, "refresh": "slow", "outline": "sam2"}
        backend.apply_model_settings(values)
        self.assertIsNone(pipeline.query_detector)
        self.assertIsNone(pipeline.segmenter)
        self.assertEqual(pipeline.requery_s, 3.)
        self.assertIsNone(pipeline.last_query_submit_s)
        self.assertFalse(state.durations_ms)
        self.assertIs(pipeline.core.mapping, mapping)
        self.assertEqual(backend.model_settings["outline"], "sam2")


class WorkspaceModelActionTests(unittest.TestCase):
    def test_demo_saves_settings_for_the_next_connection(self):
        app = WebControl(demo=True)
        try:
            workspace = app.workspace
            workspace.initialization.result(5)
            workspace.submit('models', {'settings': {'finder': 'fast'}}).result(5)
            state = workspace.snapshot()
            self.assertEqual(state['model_settings']['finder'], 'fast')
            self.assertIn('apply when connected', state['message'])
            with self.assertRaisesRegex(ValueError, 'Unknown choice'):
                workspace.submit('models', {'settings': {'finder': 'magic'}}).result(5)
        finally:
            app.close()

    def test_live_change_clears_target_approval_and_preloads(self):
        interaction = Mock()
        backend = Mock(mode='SPOT_SENSORS_DRY_RUN')
        workspace = SimpleNamespace(epoch=0, revision=0, navigation_controller=None,
            backend=backend, interaction=interaction, message='',
            model_settings=dict(model_settings.DEFAULTS), _preload_models=Mock())
        Workspace._action(workspace, 'models', {'settings': {'outline': 'sam2'}}, 0)
        backend.apply_model_settings.assert_called_once()
        self.assertEqual(backend.apply_model_settings.call_args[0][0]['outline'], 'sam2')
        interaction.invalidate.assert_called_once()
        workspace._preload_models.assert_called_once()

    def test_changes_are_refused_during_go(self):
        workspace = SimpleNamespace(epoch=0, navigation_controller='tab', backend=Mock())
        with self.assertRaisesRegex(ValueError, 'Stop the current GO'):
            Workspace._action(workspace, 'models', {'settings': {}}, 0)
        workspace.backend.apply_model_settings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
