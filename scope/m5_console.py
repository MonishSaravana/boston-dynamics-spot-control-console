"""M5 interaction console. Demo is synthetic; Spot starts in sensor-only dry-run."""
import argparse
import argparse
from dataclasses import replace
import json
import math
import sys
import time

from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw

from .live_interaction import InteractionError, InteractionSession, RobotPose
from .runtime import ModuleConfig

def _qimage(picture):
    pixels = picture.convert("RGB")
    raw = pixels.tobytes()
    return QtGui.QImage(raw, pixels.width, pixels.height, pixels.width*3,
                        QtGui.QImage.Format.Format_RGB888).copy()


# Both consoles use the same adapters and destination/authorization pipeline.
from .console_backend import (DemoBackend, SpotBackend, world_picture, camera_box,
                              draw_perception_overlay, _demo_map, _demo_entities)


class InteractionWindow(QtWidgets.QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.session = InteractionSession(backend.executor)
        self.last_query_key = None
        self.setWindowTitle("SCOPE · target interaction · " + backend.mode)
        self.resize(1370, 860)
        self.setMinimumSize(1000, 650)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main = QtWidgets.QVBoxLayout(root)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel(f"SCOPE  ·  {backend.mode}"), 1)
        self.stop = QtWidgets.QPushButton("STOP MOVEMENT")
        self.stop.clicked.connect(self.stop_clicked)
        head.addWidget(self.stop)
        main.addLayout(head)
        safety = QtWidgets.QLabel("STOP MOVEMENT requests zero velocity. Keep the separate class E-stop available.")
        main.addWidget(safety)
        self.health = QtWidgets.QLabel()
        main.addWidget(self.health)
        splitter = QtWidgets.QSplitter()
        main.addWidget(splitter, 1)
        visual = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(visual)
        vl.addWidget(QtWidgets.QLabel("Camera · highlighted target"))
        self.camera = QtWidgets.QLabel()
        self.camera.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.camera.setMinimumSize(320, 250)
        self.camera.setStyleSheet("background:#111e26")
        vl.addWidget(self.camera, 1)
        vl.addWidget(QtWidgets.QLabel("World · observed free / occupied / unknown"))
        self.world = QtWidgets.QLabel()
        self.world.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.world.setMinimumSize(320, 250)
        self.world.setStyleSheet("background:#111e26")
        vl.addWidget(self.world, 1)
        splitter.addWidget(visual)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        self.status = QtWidgets.QLabel("Select a target")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.query = QtWidgets.QLineEdit()
        self.query.setPlaceholderText("Object phrase, e.g. chair")
        self.query.returnPressed.connect(self.find)
        row.addWidget(self.query, 1)
        find = QtWidgets.QPushButton("Find object")
        find.clicked.connect(self.find)
        row.addWidget(find)
        rl.addLayout(row)
        self.point = QtWidgets.QPushButton("Simulate pointing" if backend.mode == "SIMULATED" else "Use current pointing")
        self.point.clicked.connect(self.point_clicked)
        rl.addWidget(self.point)
        self.entity_list = QtWidgets.QListWidget()
        self.entity_list.setMinimumHeight(110)
        self.entity_list.setMaximumHeight(180)
        self.entity_list.itemClicked.connect(self.entity_clicked)
        rl.addWidget(self.entity_list, 1)
        self.candidate = QtWidgets.QLabel("No candidate")
        self.candidate.setWordWrap(True)
        rl.addWidget(self.candidate)
        buttons = QtWidgets.QHBoxLayout()
        self.confirm = QtWidgets.QPushButton("CONFIRM TARGET")
        self.confirm.clicked.connect(self.confirm_clicked)
        buttons.addWidget(self.confirm)
        reject = QtWidgets.QPushButton("REJECT")
        reject.clicked.connect(self.reject_clicked)
        buttons.addWidget(reject)
        rl.addLayout(buttons)
        self.destination = QtWidgets.QLabel("No destination")
        self.destination.setWordWrap(True)
        rl.addWidget(self.destination)
        self.go = QtWidgets.QPushButton("GO · virtual" if backend.mode == "SIMULATED" else
                                        "GO · Spot (supervised)" if backend.mode == "SPOT_SUPERVISED" else
                                        "GO · DRY RUN")
        self.go.clicked.connect(self.go_clicked)
        rl.addWidget(self.go)
        self.camera_panel = QtWidgets.QGroupBox("Camera sources")
        self.camera_layout = QtWidgets.QVBoxLayout(self.camera_panel)
        self.camera_rows = {}
        self._build_camera_rows()
        rl.addWidget(self.camera_panel)
        self.dev = QtWidgets.QGroupBox("Developer telemetry")
        self.dev.setCheckable(True)
        self.dev.setChecked(False)
        dl = QtWidgets.QVBoxLayout(self.dev)
        self.telemetry = QtWidgets.QPlainTextEdit()
        self.telemetry.setReadOnly(True)
        self.telemetry.setMaximumHeight(120)
        dl.addWidget(self.telemetry)
        self.telemetry.setVisible(False)
        self.dev.toggled.connect(self.telemetry.setVisible)
        rl.addWidget(self.dev)
        rl.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([760, 610])
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(200)
        self.setStyleSheet("QWidget{font-size:14px;background:#14212b;color:#e1e9eb} "
            "QPushButton,QLineEdit,QListWidget,QPlainTextEdit,QGroupBox{padding:7px;"
            "border:1px solid #536b72;border-radius:4px} "
            "QPushButton{background:#29424a} QPushButton:hover{background:#38616a} "
            "QPushButton:disabled{color:#748b91} "
            "QPushButton#go{background:#357357} QPushButton#stop{background:#a13d3d}")
        self.go.setObjectName("go")
        self.stop.setObjectName("stop")
        self.tick()

    def _build_camera_rows(self):
        for name, status in self.backend.camera_status.items():
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(name), 1)
            widgets = []
            for label, key in (("Acquire", "acquire"), ("Process", "process"), ("Display", "display")):
                box = QtWidgets.QCheckBox(label)
                box.setChecked(status.get({"acquire":"acquired","process":"processed","display":"displayed"}[key], True))
                if self.backend.mode != "SIMULATED":
                    box.toggled.connect(lambda checked, n=name, k=key:
                        self.backend.source.configure(n, **{k: checked}))
                else:
                    box.setEnabled(False)
                row.addWidget(box)
                widgets.append(box)
            rate = QtWidgets.QDoubleSpinBox()
            rate.setRange(.2, 30.)
            rate.setSuffix(" Hz")
            rate.setValue(status["max_hz"])
            if self.backend.mode != "SIMULATED":
                rate.valueChanged.connect(lambda value, n=name:
                    self.backend.source.configure(n, max_hz=value))
            else:
                rate.setEnabled(False)
            row.addWidget(rate)
            health = QtWidgets.QLabel(status["health"])
            row.addWidget(health)
            self.camera_layout.addLayout(row)
            self.camera_rows[name] = health

    def _report(self, func):
        try:
            result = func()
            self.status.setText(self.session.message)
            return result
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return None
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Interaction action failed")
            self.status.setText(f"FAILED · {type(exc).__name__}: {exc}")
            return None

    def find(self):
        def operation():
            result = self.backend.query(self.query.text())
            if self.backend.mode == "SIMULATED":
                self.session.offer(result, ambiguous=len(result)>1)
            else:
                self.session.offer([])
                self.session.message = "PENDING · local visual query"
            return result
        self._report(operation)
        self.render()

    def point_clicked(self):
        def operation():
            candidates, unresolved = self.backend.pointing()
            if unresolved:
                self.session.offer([], ambiguous=True)
                self.session.message = "POINTING UNRESOLVED"
            else:
                self.session.offer(candidates)
            return candidates
        self._report(operation)
        self.render()

    def entity_clicked(self, item):
        entity_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        def operation():
            if any(c.entity_id == entity_id for c in self.session.candidates):
                self.session.select(entity_id)
            else:
                c = self.backend.direct(entity_id)
                self.session.offer([c])
        self._report(operation)
        self.render()

    def confirm_clicked(self):
        self._report(lambda: self.session.confirm())
        self.render()
        if self.session.confirmed:
            self._report(lambda: self.session.propose(self.backend.robot, self.backend.mapping))
            self.render()

    def reject_clicked(self):
        self._report(lambda: self.session.reject())
        self.render()

    def go_clicked(self):
        def operation():
            result = self.session.go(self.backend.robot, self.backend.mapping,
                                     self.backend.ready_for_go)
            if self.backend.mode == "SIMULATED" and self.backend.executor.pose:
                x, y, yaw = self.backend.executor.pose
                self.backend.robot = RobotPose(x, y, yaw, time.monotonic(),
                                               self.backend.world_frame)
            return result
        self._report(operation)
        self.render()

    def stop_clicked(self):
        self.session.stop()
        self.render()

    def tick(self):
        try:
            self.backend.tick()
            if self.backend.mode == "SIMULATED" and self.session.selected:
                old = self.session.selected
                self.session.refresh_observation(replace(old,
                    observed_host_s=time.monotonic(),
                    source_timestamp_s=self.backend.frame.timestamp_s))
            if self.backend.mode != "SIMULATED" and self.session.selected:
                tracked = self.backend.tracked_candidate(self.session.selected)
                if tracked:
                    self.session.refresh_observation(tracked)
            execution = self.backend.execution_status() if hasattr(self.backend, "execution_status") else None
            if execution and execution not in ("IDLE", "MOVING; measured feedback pending"):
                self.status.setText(execution)
            if self.backend.mode != "SIMULATED":
                candidates = self.backend.query_candidates()
                if candidates is not None:
                    key = tuple((c.entity_id, c.source_timestamp_s, c.score) for c in candidates)
                    if key != self.last_query_key:
                        self.last_query_key = key
                        if self.session.confirmed:
                            refreshed = next((c for c in candidates if
                                c.entity_id == self.session.confirmed.candidate.entity_id), None)
                            if refreshed is None:
                                self.session.invalidate("TARGET LOST or query stale")
                            else:
                                self.session.refresh_observation(refreshed)
                        else:
                            self.session.offer(candidates, ambiguous=len(candidates)>1)
            self.render()
        except Exception as exc:
            self.status.setText(f"DEGRADED · {type(exc).__name__}: {exc}")

    def render(self):
        if self.backend.mode == "SIMULATED":
            self.health.setText("Spot SIMULATED  ·  Cameras SIMULATED  ·  Mapping SIMULATED  ·  Objects SIMULATED  ·  Humans DISABLED  ·  Network OFFLINE")
        else:
            statuses = self.backend.camera_status
            cameras = "OK" if any(s["health"] == "OK" for s in statuses.values()) else "UNAVAILABLE"
            mapping = "OK" if self.backend.mapping is not None and self.backend.mapping.revision else "UNAVAILABLE"
            objects = "OK" if self.backend.entities else "UNAVAILABLE"
            humans = "DISABLED" if not self.backend.human_pose else (
                "OK" if self.backend.pipeline and self.backend.pipeline.core_snapshot and
                self.backend.pipeline.core_snapshot.get("tracks") else "UNAVAILABLE")
            spot = "OK" if self.backend.robot else "STALE"
            network = "DEGRADED" if any(s["health"] == "FAILED" for s in statuses.values()) else cameras
            self.health.setText(f"Spot {spot}  ·  Cameras {cameras}  ·  Mapping {mapping}  ·  "
                                f"Objects {objects}  ·  Humans {humans}  ·  Network {network}")
        frame = self.backend.frame
        visible = (frame is not None and (self.backend.mode == "SIMULATED" or
                   self.backend.source.policy[self.backend.visual_source].display))
        if visible:
            image = Image.fromarray(frame.rgb)
            draw_perception_overlay(image, frame, self.backend)
            draw = ImageDraw.Draw(image)
            c = self.session.selected
            box = camera_box(c, frame) if c else None
            if c and box:
                draw.rectangle(box, outline="#ffe36b", width=4)
                draw.text((box[0], box[1]), c.entity_id, fill="#ffe36b")
            self.camera.setPixmap(QtGui.QPixmap.fromImage(_qimage(image)).scaled(
                self.camera.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
            if self.backend.mode != "SIMULATED":
                visual = self.backend.source.latest.get(self.backend.visual_source)
                if visual:
                    self.backend.source.mark_displayed(self.backend.visual_source,
                                                       visual.acquisition_robot_s)
        else:
            self.camera.clear()
            self.camera.setText("Camera display disabled or unavailable")
        world = world_picture(self.backend.mapping, self.backend.entities,
                              self.backend.robot, self.session.proposal,
                              self.session.selected)
        self.world.setPixmap(QtGui.QPixmap.fromImage(_qimage(world)).scaled(
            self.world.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        offered = {c.entity_id: c for c in self.session.candidates}
        records = [(c.entity_id, f"Candidate · {c.entity_id} · {c.score:.2f}")
                   for c in self.session.candidates]
        records += [(e["entity_id"], f'Entity · {e["entity_id"]} · {e["label"]}')
                    for e in self.backend.entities if e["entity_id"] not in offered]
        current = [(self.entity_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole),
                    self.entity_list.item(i).text()) for i in range(self.entity_list.count())]
        if current != records:
            self.entity_list.clear()
            for entity_id, label in records:
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, entity_id)
                self.entity_list.addItem(item)
        c = self.session.selected
        self.candidate.setText(
            "No candidate" if c is None else
            f"Candidate: {c.entity_id} · {c.source} · evidence {c.score:.2f}\n"
            f"camera: {c.camera or '—'} · age: {time.monotonic()-c.observed_host_s:.1f} s "
            f"· geometry: {'yes' if c.bounds_low_m else 'unavailable'}")
        p = self.session.proposal
        self.destination.setText(
            "No destination" if p is None else
            f"Destination preview: ({p.x_m:.2f}, {p.y_m:.2f}) m · yaw {math.degrees(p.yaw_rad):.0f}°\n"
            f"stand-off {p.stand_off_m:.2f} m · clearance {p.clearance_m:.2f} m "
            f"· {len(p.route_xy_m)} observed-free route samples")
        now = time.monotonic()
        fresh_target = c is not None and 0 <= now-c.observed_host_s <= 1.5
        self.confirm.setEnabled(fresh_target and self.session.confirmed is None)
        self.go.setEnabled(p is not None and fresh_target and
                           0 <= now-p.proposed_host_s <= 10.)
        for name, label in self.camera_rows.items():
            status = self.backend.camera_status[name]
            hz = status.get("observed_acquire_hz")
            label.setText(status["health"] + (f" {hz:.1f} Hz" if hz is not None else ""))
        lines = [f"{k}: {v:.1f} ms" for k, v in self.backend.timings.items()]
        lines += [f"{k}: current {v['current_ms']:.1f} / mean {v['mean_ms']:.1f} / "
                  f"median {v['median_ms']:.1f} / p95 {v['p95_ms']:.1f} ms"
                  for k, v in self.session.timings.snapshot().items()]
        if self.backend.mode != "SIMULATED":
            lines += [f"{k}: {v}" for k, v in self.backend.source.timings.snapshot().items()]
            lines += [f"{name}: acquired {s['acquired_count']} / processed {s['processed_count']} / "
                      f"displayed {s['displayed_count']} · age {s['age_s']:.2f} s"
                      for name, s in self.backend.camera_status.items() if s['age_s'] is not None]
            if self.backend.snapshot:
                stages = ("mapping", "open_vocabulary", "object_tracking", "human_pose",
                          "pointing", "intersection", "query_projection")
                for name in stages:
                    stage = self.backend.snapshot["telemetry"].get(name)
                    if stage and stage.get("latency", {}).get("count"):
                        latency = stage["latency"]
                        lines.append(f"{name} {stage['status']}: current {stage['current_latency_ms']:.1f} / "
                            f"mean {latency['mean_ms']:.1f} / median {latency['median_ms']:.1f} / "
                            f"p95 {latency['p95_ms']:.1f} ms")
            if self.backend.error:
                lines.append(self.backend.error)
            if self.backend.mode == "SPOT_SUPERVISED":
                navigation = self.backend.executor.trajectory
                lines += [f"navigation/{k}: current {v['current_ms']:.1f} / mean {v['mean_ms']:.1f} / "
                          f"median {v['median_ms']:.1f} / p95 {v['p95_ms']:.1f} ms"
                          for k, v in navigation.timings.snapshot().items()]
                lines.append(f"measured odom samples: {len(navigation.measured)}")
        self.telemetry.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.timer.stop()
        self.session.stop()
        self.backend.close()
        event.accept()

    def changeEvent(self, event):
        if (event.type() == QtCore.QEvent.Type.WindowDeactivate and
                self.backend.mode == "SPOT_SUPERVISED"):
            self.stop_clicked()
        super().changeEvent(event)

    def hideEvent(self, event):
        if self.backend.mode == "SPOT_SUPERVISED":
            self.stop_clicked()
        super().hideEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic room and virtual robot")
    mode.add_argument("--spot", metavar="HOST", help="Read-only sensors and dry-run GO")
    parser.add_argument("--visual-source", help="Listed Spot visual source")
    parser.add_argument("--depth-source", help="Listed aligned depth source")
    parser.add_argument("--alignment-verified", action="store_true",
        help="Assert that physical pixel alignment, intrinsics and transforms were checked")
    parser.add_argument("--human-pose", action="store_true", help="Enable optional pose detector")
    parser.add_argument("--supervised-go", action="store_true",
        help="After dry-run gates: acquire a lease and enable explicit GO on Spot")
    args = parser.parse_args(argv)
    if args.spot and not args.visual_source:
        parser.error("--spot needs --visual-source from 'scope spot-sensors'")
    if args.alignment_verified and not args.depth_source:
        parser.error("--alignment-verified needs --depth-source")
    if args.supervised_go and (not args.spot or not args.alignment_verified):
        parser.error("--supervised-go needs --spot and verified RGB-D geometry")
    backend = DemoBackend() if args.demo else SpotBackend(args.spot, args.visual_source,
        args.depth_source, args.alignment_verified, args.human_pose, args.supervised_go)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InteractionWindow(backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
