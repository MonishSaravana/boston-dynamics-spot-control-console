"""Projected body-offset diagram used by the reconstructed preview stage."""

import math

from bosdyn.geometry import EulerZXY
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class BodyPosturePreview(QWidget):
    """Body-frame diagram only: no legs, feet, collision, or reachability model."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(460, 300)
        self.height_cm = self.roll_deg = self.pitch_deg = 0

    def set_request(self, height_cm, roll_deg, pitch_deg):
        self.height_cm, self.roll_deg, self.pitch_deg = height_cm, roll_deg, pitch_deg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#101820'))
        w, h = self.width(), self.height()
        center = QPointF(w * .5, h * .55)
        scale = min(w / 4.5, h / 3.0)

        def project(v):
            x, y, z = v
            return QPointF(center.x() + scale * (.78 * x - .58 * y),
                           center.y() + scale * (.30 * x + .32 * y - z))

        verts = [(x, y, z) for z in (-.22, .22)
                 for y in (-.48, .48) for x in (-.9, .9)]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if sum(verts[a][i] != verts[b][i] for i in range(3)) == 1]

        def draw_box(points, color, dashed=False):
            pen = QPen(QColor(color), 2)
            if dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for a, b in edges:
                painter.drawLine(project(points[a]), project(points[b]))

        draw_box(verts, '#58636b', True)
        q = EulerZXY(roll=math.radians(self.roll_deg),
                     pitch=math.radians(self.pitch_deg)).to_quaternion()

        def rotate(v):
            x, y, z = v
            return ((1 - 2 * (q.y*q.y + q.z*q.z))*x +
                    2 * (q.x*q.y - q.z*q.w)*y + 2 * (q.x*q.z + q.y*q.w)*z,
                    2 * (q.x*q.y + q.z*q.w)*x +
                    (1 - 2 * (q.x*q.x + q.z*q.z))*y + 2 * (q.y*q.z - q.x*q.w)*z,
                    2 * (q.x*q.z - q.y*q.w)*x + 2 * (q.y*q.z + q.x*q.w)*y +
                    (1 - 2 * (q.x*q.x + q.y*q.y))*z + self.height_cm / 10)

        draw_box([rotate(v) for v in verts], '#56d7d0')
        painter.setPen(QColor('#d9e4e8'))
        painter.drawText(12, 22, 'DESCRIPTIVE BODY-FRAME PREVIEW')
        painter.drawText(12, h - 34, 'Gray: nominal   Cyan: requested offset')
        painter.drawText(12, h - 15, 'Not to scale • no legs or feet • not a validated pose')
        painter.end()
