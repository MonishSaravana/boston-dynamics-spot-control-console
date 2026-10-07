"""Dark Qt theme shared by the desktop consoles; tokens match web/style.css."""

COLORS = {
    "bg": "#0b0c0d",
    "panel": "#111314",
    "raised": "#181b1d",
    "hover": "#202427",
    "canvas": "#070808",
    "line": "#23272a",
    "line_strong": "#343a3e",
    "text": "#e6e8e9",
    "muted": "#9ba3a8",
    "subtle": "#848d92",
    "accent": "#3ecf9b",
    "accent_hover": "#62dcb0",
    "accent_ink": "#04140e",
    "target": "#f0b44c",
    "stop": "#c42f39",
    "stop_hover": "#d83a45",
    "error": "#ff8f98",
    "map_unknown": "#070808",
    "map_free": "#16191b",
    "map_occupied": "#3b4347",
}

FONT = '-apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial'
MONO = '"SF Mono", Menlo, Consolas, monospace'

STYLE = """
QMainWindow, QWidget#root {{ background: {bg}; color: {text}; }}
QWidget {{ color: {text}; font-family: {font}; font-size: 13px; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {line_strong}; padding: 5px; }}

QFrame#topbar {{ background: {bg}; border: none; border-bottom: 1px solid {line}; }}
QLabel#brand {{ font-size: 15px; font-weight: 600; letter-spacing: 3px; }}
QLabel#navCurrent {{ color: {text}; font-weight: 600; }}
QFrame#navLine {{ background: {accent}; border: none; }}
QLabel#modePill {{ color: {text}; border: 1px solid {line_strong}; border-radius: 3px;
                   padding: 5px 10px; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#stopNote {{ color: {muted}; font-size: 11px; }}
QPushButton#stopButton {{ background: {stop}; border: 1px solid {stop}; color: white;
                          font-weight: 700; letter-spacing: 1px; padding: 8px 16px; border-radius: 3px; }}
QPushButton#stopButton:hover {{ background: {stop_hover}; }}
QPushButton#stopButton:focus {{ border: 1px solid {text}; }}

QFrame#statusStrip, QFrame#footer {{ background: {panel}; border: none; border-bottom: 1px solid {line}; }}
QFrame#footer {{ border-bottom: none; border-top: 1px solid {line}; }}
QLabel#stateTitle {{ font-weight: 600; }}
QLabel#muted, QLabel#cameraStatus, QLabel#footerText {{ color: {muted}; font-size: 12px; }}
QLabel#sectionTitle {{ color: {text}; font-size: 13px; font-weight: 600; padding-top: 4px; }}
QLabel#panelTitle {{ color: {text}; font-weight: 600; }}
QLabel#mono {{ font-family: {mono}; color: {muted}; font-size: 12px; }}
QLabel#targetText {{ font-family: {mono}; color: {target}; }}
QLabel#badge {{ color: {muted}; border: 1px solid {line_strong}; border-radius: 3px;
                padding: 2px 6px; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}

QFrame#surface, QFrame#controlPanel, QWidget#panel {{ background: {panel}; border: none; }}
QFrame#canvas {{ background: {canvas}; border: none; }}
QFrame#rule {{ background: {line}; max-height: 1px; border: none; }}

QDockWidget {{ color: {text}; font-weight: 600; }}
QDockWidget::title {{ background: {panel}; padding: 8px 10px; border-bottom: 1px solid {line}; }}
QMainWindow::separator {{ background: {line}; width: 1px; height: 1px; }}
QSplitter::handle {{ background: {line}; }}

QPushButton, QToolButton {{ background: {raised}; color: {text}; border: 1px solid {line_strong};
                            border-radius: 5px; padding: 7px 12px; font-weight: 500; }}
QPushButton:hover, QToolButton:hover {{ background: {hover}; }}
QPushButton:focus, QToolButton:focus {{ border: 1px solid {accent}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {subtle}; background: {panel}; border-color: {line}; }}
QToolButton#quiet {{ background: transparent; border: 1px solid transparent; color: {muted}; }}
QToolButton#quiet:hover {{ color: {text}; background: {hover}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QPushButton#goButton, QPushButton#applyButton {{ background: {accent}; border-color: {accent};
                                                 color: {accent_ink}; font-weight: 700; }}
QPushButton#goButton {{ padding: 11px 12px; letter-spacing: 1px; }}
QPushButton#goButton:hover, QPushButton#applyButton:hover {{ background: {accent_hover}; }}
QPushButton#goButton:disabled, QPushButton#applyButton:disabled {{ background: {raised};
                                                 border-color: {line}; color: {subtle}; }}
QPushButton#confirmButton {{ border-color: {accent}; color: {accent}; }}
QPushButton#confirmButton:disabled {{ border-color: {line}; color: {subtle}; }}

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QListWidget {{
    background: {canvas}; color: {text}; border: 1px solid {line_strong}; border-radius: 5px;
    padding: 6px 8px; selection-background-color: {accent}; selection-color: {accent_ink}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
QComboBox QAbstractItemView {{ background: {raised}; color: {text}; border: 1px solid {line_strong};
                               selection-background-color: {hover}; selection-color: {text}; }}
QListWidget::item {{ padding: 5px 4px; border-bottom: 1px solid {line}; }}
QListWidget::item:selected {{ background: {hover}; color: {target}; }}
QGroupBox {{ border: 1px solid {line}; border-radius: 5px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {muted}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator, QGroupBox::indicator {{ width: 14px; height: 14px; border: 1px solid {line_strong};
                                              border-radius: 3px; background: {canvas}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QCheckBox:disabled {{ color: {subtle}; }}

QSlider {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: {line_strong}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; background: {text};
                              border: 1px solid {bg}; border-radius: 7px; }}
QSlider::sub-page:horizontal:disabled {{ background: {line_strong}; }}

QTabWidget::pane {{ border: none; background: {panel}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {panel}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {line_strong}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: {panel}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {line_strong}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QMenu {{ background: {raised}; border: 1px solid {line_strong}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; }}
QMenu::item:selected {{ background: {hover}; }}
QMenu::separator {{ height: 1px; background: {line}; margin: 4px 0; }}
""".format(font=FONT, mono=MONO, **COLORS)


def nav_label(text):
    """Current-page label with the browser's accent underline."""
    from PySide6 import QtWidgets
    box = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(0, 6, 0, 0)
    layout.setSpacing(5)
    label = QtWidgets.QLabel(text)
    label.setObjectName("navCurrent")
    layout.addWidget(label)
    line = QtWidgets.QFrame()
    line.setObjectName("navLine")
    line.setFixedHeight(2)
    layout.addWidget(line)
    return box
