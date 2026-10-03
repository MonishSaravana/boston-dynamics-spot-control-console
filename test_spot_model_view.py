"""Offline checks of the read-only SDK URDF view."""

import unittest
from types import SimpleNamespace
from zipfile import ZipFile

from PySide6.QtWidgets import QApplication

from spot_model_view import (JOINT_NAMES, SDK_URDF, SpotModelView, SpotUrdfMesh,
                             hardware_matches_base)


class SpotModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_sdk_mesh_and_kinematic_chain(self):
        mesh = SpotUrdfMesh()
        self.assertEqual(len(mesh.meshes), 13)
        self.assertEqual(len(mesh.joints), 12)
        zero = {name: 0.0 for name in JOINT_NAMES}
        poses = mesh.link_transforms(zero)
        changed = dict(zero, **{'fl.hx': .2})
        other = mesh.link_transforms(changed)
        self.assertFalse((poses['fl.hip'] == other['fl.hip']).all())
        self.assertTrue((poses['fr.hip'] == other['fr.hip']).all())

    def test_demo_reference_is_labeled_and_read_only(self):
        view = SpotModelView()
        view.show_demo_reference()
        self.assertIn('not measured', view.message)
        view.set_measured_state(None, 'No fresh telemetry')
        self.assertIsNone(view.angles)
        view.close()

    def test_actual_skeleton_must_match_local_base_mesh(self):
        with ZipFile(SDK_URDF) as archive:
            urdf = archive.read('model.urdf').decode()
        config = SimpleNamespace(skeleton=SimpleNamespace(urdf=urdf))
        self.assertTrue(hardware_matches_base(config))
        config.skeleton.urdf = urdf.replace('</robot>', '<link name="arm"/></robot>')
        self.assertFalse(hardware_matches_base(config))


if __name__ == '__main__':
    unittest.main()
