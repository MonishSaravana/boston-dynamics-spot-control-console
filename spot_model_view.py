"""Read-only rendering of the base Spot URDF supplied with the local SDK checkout.

No joint command is constructed here. Meshes remain in the SDK ZIP and are loaded
in memory for display; the model is not a collision or balance simulation.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


JOINT_NAMES = tuple(f'{leg}.{joint}' for leg in ('fl', 'fr', 'hl', 'hr')
                    for joint in ('hx', 'hy', 'kn'))
SDK_URDF = Path(__file__).resolve().parent / 'spot-sdk/files/spot_base_urdf.zip'


def hardware_matches_base(hardware_configuration):
    """Render the local base mesh only for an exactly matching robot skeleton."""
    try:
        root = ET.fromstring(hardware_configuration.skeleton.urdf)
        with ZipFile(SDK_URDF) as archive:
            expected = ET.fromstring(archive.read('model.urdf'))
        links = {link.get('name') for link in root.findall('link')}
        joints = {joint.get('name') for joint in root.findall('joint')}
        if links != {link.get('name') for link in expected.findall('link')} or \
                joints != set(JOINT_NAMES):
            return False
        actual_joints = {joint.get('name'): joint for joint in root.findall('joint')}
        for reference in expected.findall('joint'):
            actual = actual_joints[reference.get('name')]
            if any(actual.find(part).get('link') != reference.find(part).get('link')
                   for part in ('parent', 'child')):
                return False
            for part, attr in (('origin', 'xyz'), ('origin', 'rpy'), ('axis', 'xyz')):
                lhs = np.array([float(x) for x in actual.find(part).get(attr).split()])
                rhs = np.array([float(x) for x in reference.find(part).get(attr).split()])
                if not np.allclose(lhs, rhs, atol=1e-4, rtol=0):
                    return False
        return True
    except (AttributeError, ET.ParseError, TypeError, ValueError, KeyError, FileNotFoundError):
        return False


def _transform(xyz, rpy=(0, 0, 0)):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll),
                             math.cos(pitch), math.sin(pitch),
                             math.cos(yaw), math.sin(yaw))
    result = np.eye(4)
    result[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1 - c
    result = np.eye(4)
    result[:3, :3] = ((c + x*x*k, x*y*k - z*s, x*z*k + y*s),
                      (y*x*k + z*s, c + y*y*k, y*z*k - x*s),
                      (z*x*k - y*s, z*y*k + x*s, c + z*z*k))
    return result


class SpotUrdfMesh:
    def __init__(self, path=SDK_URDF):
        if not path.is_file():
            raise FileNotFoundError(f'SDK base URDF is missing: {path}')
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read('model.urdf'))
            if root.get('name') != 'spot':
                raise ValueError('Unexpected URDF robot name')
            self.meshes = {}
            for link in root.findall('link'):
                filename = link.find('visual/geometry/mesh').get('filename')
                vertices, faces = [], []
                for line in archive.read(filename).decode('utf-8').splitlines():
                    parts = line.split()
                    if parts and parts[0] == 'v':
                        vertices.append(tuple(float(value) for value in parts[1:4]))
                    elif parts and parts[0] == 'f':
                        ids = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                        faces.extend((ids[0], ids[i], ids[i+1]) for i in range(1, len(ids)-1))
                color = '#d5ad49' if link.get('name') == 'base' or '.uleg' in link.get('name') else '#4b5963'
                self.meshes[link.get('name')] = (np.asarray(vertices, dtype=float),
                                                  np.asarray(faces, dtype=np.int32), color)
            self.joints = {}
            for joint in root.findall('joint'):
                name = joint.get('name')
                if name not in JOINT_NAMES:
                    continue
                origin = joint.find('origin')
                xyz = [float(x) for x in origin.get('xyz').split()]
                rpy = [float(x) for x in origin.get('rpy').split()]
                axis = [float(x) for x in joint.find('axis').get('xyz').split()]
                self.joints[name] = (joint.find('parent').get('link'),
                                     joint.find('child').get('link'),
                                     _transform(xyz, rpy), axis)
        if len(self.joints) != 12 or len(self.meshes) != 13:
            raise ValueError('The SDK base URDF does not contain the expected 12 leg joints')

    def link_transforms(self, angles):
        transforms = {'base': np.eye(4)}
        for name in JOINT_NAMES:
            parent, child, origin, axis = self.joints[name]
            transforms[child] = transforms[parent] @ origin @ _axis_rotation(axis, angles[name])
        return transforms


class SpotModelView(QWidget):
    """Software-rendered SDK mesh; drag or wheel only change the viewpoint."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 190)
        self.setMouseTracking(True)
        self.setToolTip('SDK base Spot URDF mesh. Left-drag to orbit, right-drag to pan, wheel to zoom. '
                        'Viewing never sends a robot command.')
        self.azimuth = math.radians(-55)
        self.elevation = math.radians(25)
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.last_mouse = None
        self.mouse_button = None
        self.angles = None
        self.message = 'Waiting for robot telemetry'
        self.mesh = None
        self.error = None
        try:
            self.mesh = SpotUrdfMesh()
        except Exception as exc:
            self.error = str(exc)
            self.message = 'SDK model unavailable'

    def set_measured_state(self, angles, message):
        if angles is None:
            self.angles = None
        elif all(name in angles and math.isfinite(angles[name]) for name in JOINT_NAMES):
            self.angles = dict(angles)
        else:
            self.angles = None
        self.message = message
        self.update()

    def show_demo_reference(self):
        self.angles = {name: 0.0 for name in JOINT_NAMES}
        self.message = 'SIMULATED • URDF zero configuration, not measured robot state'
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.last_mouse = event.position()
            self.mouse_button = event.button()

    def mouseMoveEvent(self, event):
        if self.last_mouse is not None:
            delta = event.position() - self.last_mouse
            if self.mouse_button == Qt.LeftButton:
                self.azimuth += delta.x() * 0.008
                self.elevation = max(math.radians(-10), min(math.radians(85),
                                      self.elevation + delta.y() * 0.008))
            else:
                self.pan += np.array([delta.x(), delta.y()])
            self.last_mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse = None
        self.mouse_button = None

    def wheelEvent(self, event):
        self.zoom = max(0.5, min(2.5, self.zoom * (1.1 if event.angleDelta().y() > 0 else 1/1.1)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor('#070808'))
        width, height = self.width(), self.height()
        az, el = self.azimuth, self.elevation
        right = np.array([-math.sin(az), math.cos(az), 0.0])
        up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
        toward_camera = np.array([math.cos(el)*math.cos(az),
                                  math.cos(el)*math.sin(az), math.sin(el)])
        transformed = {}
        if self.mesh is not None and self.angles is not None:
            transforms = self.mesh.link_transforms(self.angles)
            for link, (vertices, _, _) in self.mesh.meshes.items():
                matrix = transforms[link]
                transformed[link] = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            all_points = np.concatenate(list(transformed.values()))
            raw = np.stack((all_points @ right, all_points @ up), axis=-1)
            minimum, maximum = raw.min(axis=0), raw.max(axis=0)
            span = np.maximum(maximum - minimum, .01)
            scale = min(width * .76 / span[0], height * .70 / span[1]) * self.zoom
            midpoint = (minimum + maximum) / 2
            center = np.array([width * .5, height * .5]) - midpoint * np.array([scale, -scale])
        else:
            scale = min(width / 1.65, height / 1.35) * self.zoom
            center = np.array([width * .5, height * .53])
        center += self.pan

        def project(points):
            xy = np.stack((points @ right, points @ up), axis=-1)
            return center + xy * np.array([scale, -scale])

        # This reference grid is fixed to the URDF body frame, not a measured floor.
        painter.setPen(QPen(QColor('#24292c'), 1))
        for coordinate in np.arange(-0.8, 0.81, 0.2):
            a, b = project(np.array([[-.9, coordinate, -.53], [.9, coordinate, -.53]]))
            painter.drawLine(QPointF(*a), QPointF(*b))
            a, b = project(np.array([[coordinate, -.9, -.53], [coordinate, .9, -.53]]))
            painter.drawLine(QPointF(*a), QPointF(*b))
        for axis, color in (([.35, 0, 0], '#ed746d'), ([0, .35, 0], '#65cfa1'),
                            ([0, 0, .35], '#79b9ef')):
            a, b = project(np.array([[0, 0, 0], axis]))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(QPointF(*a), QPointF(*b))

        if self.mesh is not None and self.angles is not None:
            triangles = []
            light = np.array([.4, -.5, .75])
            light /= np.linalg.norm(light)
            for link, (vertices, faces, base_color) in self.mesh.meshes.items():
                points = transformed[link]
                face_points = points[faces]
                normals = np.cross(face_points[:, 1]-face_points[:, 0],
                                   face_points[:, 2]-face_points[:, 0])
                visible = normals @ toward_camera > 0
                face_points = face_points[visible]
                normals = normals[visible]
                if not len(face_points):
                    continue
                screen = project(face_points.reshape(-1, 3)).reshape(-1, 3, 2)
                depths = face_points.mean(axis=1) @ toward_camera
                lengths = np.linalg.norm(normals, axis=1)
                shades = np.clip((normals @ light) / np.maximum(lengths, 1e-9), 0, 1)
                for depth, polygon, shade in zip(depths, screen, shades):
                    triangles.append((float(depth), polygon, base_color, float(shade)))
            triangles.sort(key=lambda item: item[0])
            painter.setPen(Qt.NoPen)
            for _, polygon, base_color, shade in triangles:
                base = QColor(base_color)
                factor = .52 + .48 * shade
                painter.setBrush(QColor(int(base.red()*factor), int(base.green()*factor),
                                        int(base.blue()*factor)))
                painter.drawPolygon(QPolygonF([QPointF(*point) for point in polygon]))
        else:
            painter.setPen(QColor('#9ba3a8'))
            painter.drawText(self.rect().adjusted(20, 30, -20, -30),
                             Qt.AlignCenter | Qt.TextWordWrap,
                             self.error or 'Robot geometry appears when fresh joint telemetry is available')
        painter.setPen(QColor('#e6e8e9'))
        painter.drawText(self.rect().adjusted(12, 8, -12, -height + 56),
                         Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, self.message)
        painter.setPen(QColor('#848d92'))
        painter.drawText(self.rect().adjusted(12, height - 46, -12, -8),
                         Qt.AlignLeft | Qt.AlignBottom | Qt.TextWordWrap,
                         'SDK base URDF • read-only • body-frame grid, not floor contact')
        painter.end()
