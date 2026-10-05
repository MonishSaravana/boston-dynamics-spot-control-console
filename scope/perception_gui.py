"""Small offline/query console. It has no robot commands or network server."""

from dataclasses import replace

from PySide6 import QtCore,QtGui,QtWidgets

from .perception_cli import overlay


class QueryWindow(QtWidgets.QMainWindow):
    def __init__(self,pipeline,driver,metadata,ticks):
        super().__init__()
        self.pipeline,self.driver,self.metadata,self.ticks=pipeline,driver,metadata,ticks
        self.last_render_s=0.
        self.render_durations=[]
        self.setWindowTitle("SCOPE perception · local object queries")
        self.resize(1120,780)
        self.setMinimumSize(850,600)
        root=QtWidgets.QWidget();self.setCentralWidget(root)
        layout=QtWidgets.QVBoxLayout(root)
        title=QtWidgets.QLabel("Object query")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        input_row=QtWidgets.QHBoxLayout()
        self.phrase=QtWidgets.QLineEdit();self.phrase.setPlaceholderText("gray sofa, backpack, power strip…")
        self.phrase.setAccessibleName("Object phrase")
        input_row.addWidget(self.phrase,1)
        button=QtWidgets.QPushButton("Find object");button.setAccessibleName("Find object")
        button.clicked.connect(self.request);self.phrase.returnPressed.connect(self.request)
        input_row.addWidget(button);layout.addLayout(input_row)
        self.status=QtWidgets.QLabel("Enter an ordinary object phrase. Model scores are uncalibrated evidence.")
        self.status.setWordWrap(True);self.status.setMinimumHeight(42)
        layout.addWidget(self.status)
        flags=QtWidgets.QHBoxLayout()
        for label,name in (("Queries","open_vocabulary"),("Common objects","object_detector"),
                           ("Human pose","human_pose"),("Mapping","mapping"),("Memory","query_memory"),("Rerun","viewer")):
            checkbox=QtWidgets.QCheckBox(label);checkbox.setAccessibleName(label)
            checkbox.setChecked(pipeline.runtime.state(name).config.enabled)
            checkbox.toggled.connect(lambda enabled,n=name:pipeline.configure(n,
                replace(pipeline.runtime.state(n).config,enabled=enabled)))
            flags.addWidget(checkbox)
        flags.addStretch();layout.addLayout(flags)
        split=QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.image=QtWidgets.QLabel();self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(400,300);self.image.setStyleSheet("background:#111820;")
        split.addWidget(self.image)
        right=QtWidgets.QWidget();right_layout=QtWidgets.QVBoxLayout(right)
        self.evidence=QtWidgets.QPlainTextEdit();self.evidence.setReadOnly(True)
        self.evidence.setAccessibleName("Query evidence");self.evidence.setMinimumHeight(140)
        right_layout.addWidget(self.evidence,1)
        self.telemetry=QtWidgets.QTableWidget(0,13)
        self.telemetry.setHorizontalHeaderLabels(["Module","State","Hz","Age s","Last ms","p95 ms",
            "Mean ms","Median ms","Queue","Dropped","Backend","Device","Model"])
        self.telemetry.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.telemetry.verticalHeader().hide()
        self.telemetry.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.telemetry.horizontalHeader().setDefaultSectionSize(65)
        for column,width in enumerate((105,90,45,55,65,65)):self.telemetry.setColumnWidth(column,width)
        self.telemetry.setColumnWidth(10,110);self.telemetry.setColumnWidth(12,320)
        right_layout.addWidget(self.telemetry,2)
        self.resources=QtWidgets.QLabel();self.resources.setWordWrap(True)
        right_layout.addWidget(self.resources)
        split.addWidget(right);split.setSizes([650,450]);layout.addWidget(split,1)
        source=f'{metadata.get("scene",metadata.get("source","recording"))} · {metadata.get("world_frame","")}'
        self.source_label=QtWidgets.QLabel(source+"\n3D requires aligned depth, calibration, and a declared camera/world pose.")
        self.source_label.setWordWrap(True);layout.addWidget(self.source_label)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)
        self.setStyleSheet("QWidget { font-size: 13px; } QLineEdit { padding:8px; } QPushButton { padding:8px 16px; }")

    def request(self):
        try:
            self.pipeline.request(self.phrase.text())
            self.status.setText("PENDING · Searching the current image. First model startup can take several seconds.")
        except ValueError as exc:
            self.status.setText(str(exc))

    def tick(self):
        import time
        try:
            frame=self.driver.poll(self.pipeline.runtime);start=time.perf_counter()
            s=self.pipeline.tick(frame,self.driver.now())
            self.ticks.append((time.perf_counter()-start)*1000)
            if time.monotonic()-self.last_render_s<.2:
                return
            self.last_render_s=time.monotonic()
            render_start=time.perf_counter()
            rgb=overlay(frame,s)
            pixels=rgb.tobytes()
            image=QtGui.QImage(pixels,rgb.width,rgb.height,rgb.width*3,QtGui.QImage.Format.Format_RGB888).copy()
            self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.image.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))
            lines=[]
            for q in sorted(s["queries"],key=lambda q:q["normalized_query"]!=s["active_query"]):
                model=q["model"]
                lines.extend([f'{q["user_phrase"]} — {q["state"]}',
                    f'{model.get("model","").split("/")[-1]} · {model.get("device","")} · {model.get("backend","")}',q["reason"],
                    "Expanded query: "+", ".join(q["expanded_queries"]),
                    f'Input snapshot age: {q["age_s"]:.2f} s'+(" · immutable still; capture time unknown" if s["input_kind"]=="immutable still" else "")])
                if q["result_age_s"] is not None:lines.append(f'Query result received {q["result_age_s"]:.2f} s ago')
                for c in q["candidates"]:
                    lines.append(f'{c["detector_match"]}: evidence {c["score"]:.3f}; mask {c["mask_available"]}')
                    verification=c["evidence"].get("verification")
                    if verification:
                        score="—" if verification["score"] is None else f'{verification["score"]:.3f}'
                        overlap="—" if verification["box_iou"] is None else f'{verification["box_iou"]:.2f}'
                        lines.append(f'OWLv2 agreement: {verification["agreed"]}; score {score}; box IoU {overlap}')
                    if c["projection"]:
                        xyz=c["projection"]["center_m"]
                        lines.append(f'Observed center (m): {xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}')
                lines.append("")
            self.evidence.setPlainText("\n".join(lines))
            if self.pipeline.pending_query:
                result=self.pipeline.results.get(self.pipeline.pending_query[1].normalized)
                if result:self.status.setText(f'{result.state} · {result.query.raw_phrase} · {result.reason}')
            rows=[(n,t) for n,t in s["telemetry"].items() if t["successes"] or n in
                ("mapping","object_detector","human_pose","open_vocabulary","object_tracking","segmentation","viewer")]
            priority={n:i for i,n in enumerate(("open_vocabulary","object_tracking","mapping","object_detector","human_pose","viewer","segmentation"))}
            rows.sort(key=lambda row:priority.get(row[0],20))
            self.telemetry.setUpdatesEnabled(False)
            if self.telemetry.rowCount()!=len(rows):self.telemetry.setRowCount(len(rows))
            def number(v):return "—" if v is None else f"{v:.1f}"
            labels={"open_vocabulary":"query","object_detector":"common","human_pose":"humans",
                    "object_tracking":"object tracks","tracking":"people tracks","query_projection":"projection","resource_metrics":"resources"}
            for i,(name,t) in enumerate(rows):
                status="STATIC" if s["input_kind"]=="immutable still" and t["status"]=="STALE" and name!="resource_metrics" else t["status"]
                values=[labels.get(name,name),status,number(t["host_update_hz"]),number(t["data_age_s"]),
                        number(t["current_latency_ms"]),number(t["latency"]["p95_ms"]),
                        number(t["latency"]["mean_ms"]),number(t["latency"]["median_ms"]),
                        str(t["queue_depth"]),str(t["dropped_inputs"]),t["backend"] or "—",t["device"] or "—",t["model"] or "—"]
                for j,value in enumerate(values):
                    item=self.telemetry.item(i,j)
                    if item is None:self.telemetry.setItem(i,j,QtWidgets.QTableWidgetItem(value))
                    elif item.text()!=value:item.setText(value)
            self.telemetry.setUpdatesEnabled(True)
            r=s["resources"];rss=r.get("rss_bytes");accelerator=r.get("accelerator_allocated_bytes")
            self.resources.setText(f'Core tick {s["core_tick_ms"]:.1f} ms · '+
                (f'RSS {rss/1e6:.0f} MB' if rss else "RSS unavailable")+f' · CPU {number(r.get("cpu_percent"))}%\nLatest input age {number(s["observation_age_s"])} s · cameras {s["active_camera_count"]} · Rerun {"on" if s["rerun_enabled"] else "off"}\n'+
                (f'Accelerator allocation {accelerator/1e6:.0f} MB. ' if accelerator is not None else "")+
                'Utilization unavailable. Scroll telemetry for queues and models.')
            self.render_durations.append((time.perf_counter()-render_start)*1000)
            self.render_durations=self.render_durations[-10000:]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Perception console tick failed")
            self.status.setText(f'FAILED · {type(exc).__name__}: {exc}')

    def closeEvent(self,event):
        self.timer.stop();event.accept()


def run_window(pipeline,driver,metadata,phrase,ticks):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window=QueryWindow(pipeline,driver,metadata,ticks)
    if phrase:
        window.phrase.setText(phrase);window.request()
    window.show();app.exec()
