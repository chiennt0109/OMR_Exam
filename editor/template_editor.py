from __future__ import annotations

import copy
from pathlib import Path
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QToolBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.omr_engine import OMRProcessor, OMRResult
from core.template_engine import TemplateEngine
from models.template import AnchorPoint, Template, Zone, ZoneType

BLOCK_TYPES = {ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK, ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK}


class TemplateCanvas(QWidget):
    selection_changed = Signal(int)
    zones_changed = Signal()

    def __init__(self):
        super().__init__()
        self.template: Template | None = None
        self.pixmap: QPixmap | None = None
        self.zoom = 1.0
        self.preview_mode = False
        self.add_anchor_mode = False
        self.add_digit_anchor_mode = False
        self.current_zone_type = ZoneType.MCQ_BLOCK
        self.selected_zone = -1
        self.selected_anchor = -1

        self._drawing = False
        self._moving = False
        self._drag_corner = -1
        self._start = QPoint()
        self._current_rect = QRect()
        self._move_offset = QPoint()

        self.recognition_overlay: dict[str, list[bool]] = {}
        self.detected_anchor_points: list[tuple[float, float]] = []
        self.digit_zone_debug: dict[str, dict[str, object]] = {}
        self.setFocusPolicy(Qt.StrongFocus)

    def set_template(self, template: Template, pixmap: QPixmap):
        self.template = template
        self.pixmap = pixmap
        self.zoom = 1.0
        self.selected_zone = -1
        self.selected_anchor = -1
        self.preview_mode = False
        self.recognition_overlay.clear()
        self.detected_anchor_points = []
        self.digit_zone_debug = {}
        self.resize(int(pixmap.width() * self.zoom), int(pixmap.height() * self.zoom))
        self.selection_changed.emit(-1)
        self.update()

    def set_zoom(self, z: float):
        self.zoom = max(0.25, min(4.0, z))
        if self.pixmap:
            self.resize(int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))
        self.update()

    def _to_img(self, p: QPoint) -> QPoint:
        return QPoint(int(p.x() / self.zoom), int(p.y() / self.zoom))

    def _zone_rect_abs(self, z: Zone) -> QRect:
        assert self.template
        return QRect(int(z.x * self.template.width), int(z.y * self.template.height), int(z.width * self.template.width), int(z.height * self.template.height))

    def _control_points_abs(self, z: Zone) -> list[QPointF]:
        assert self.template
        cps = z.metadata.get("control_points")
        if not cps:
            cps = [[z.x, z.y], [z.x + z.width, z.y], [z.x, z.y + z.height], [z.x + z.width, z.y + z.height]]
            z.metadata["control_points"] = cps
        return [QPointF(c[0] * self.template.width, c[1] * self.template.height) for c in cps]

    def mousePressEvent(self, e: QMouseEvent):
        if not self.template or not self.pixmap or e.button() != Qt.LeftButton:
            return
        p = self._to_img(e.position().toPoint())
        self.setFocus()

        if self.add_anchor_mode:
            self.template.anchors.append(AnchorPoint(p.x() / self.template.width, p.y() / self.template.height, f"A{len(self.template.anchors)+1}"))
            self.zones_changed.emit(); self.update(); return
        if self.add_digit_anchor_mode:
            digit_count = len([a for a in self.template.anchors if str(getattr(a, "name", "") or "").startswith("DIGIT_ANCHOR_")])
            self.template.anchors.append(AnchorPoint(p.x() / self.template.width, p.y() / self.template.height, f"DIGIT_ANCHOR_{digit_count+1:02d}"))
            self.zones_changed.emit(); self.update(); return

        a_idx = self._hit_anchor(p)
        if a_idx >= 0:
            self.selected_anchor = a_idx
            self.selected_zone = -1
            self.selection_changed.emit(-1)
            self.update()
            return
        self.selected_anchor = -1

        z_idx, c_idx = self._hit_zone_or_control(p)
        if z_idx >= 0:
            self.selected_zone = z_idx
            self.selection_changed.emit(z_idx)
            if c_idx >= 0:
                self._drag_corner = c_idx
            else:
                self._moving = True
                zr = self._zone_rect_abs(self.template.zones[z_idx])
                self._move_offset = QPoint(p.x() - zr.x(), p.y() - zr.y())
            self.update(); return

        self.selected_zone = -1
        self.selection_changed.emit(-1)
        self._drawing = True
        self._start = p
        self._current_rect = QRect(p, p)

    def mouseMoveEvent(self, e: QMouseEvent):
        if not self.template:
            return
        p = self._to_img(e.position().toPoint())
        self.setFocus()

        if self._drawing:
            self._current_rect = QRect(self._start, p).normalized(); self.update(); return

        if self.selected_zone < 0:
            return
        z = self.template.zones[self.selected_zone]

        if self._moving:
            zr = self._zone_rect_abs(z)
            old_x, old_y = zr.x(), zr.y()
            zr.moveTo(p.x() - self._move_offset.x(), p.y() - self._move_offset.y())
            dx, dy = zr.x() - old_x, zr.y() - old_y
            z.x = zr.x() / self.template.width
            z.y = zr.y() / self.template.height
            cps = self._control_points_abs(z)
            moved = [QPointF(c.x() + dx, c.y() + dy) for c in cps]
            z.metadata["control_points"] = [[c.x() / self.template.width, c.y() / self.template.height] for c in moved]
            self.zones_changed.emit(); self.update(); return

        if self._drag_corner >= 0:
            cps = self._control_points_abs(z)
            cps[self._drag_corner] = QPointF(p.x(), p.y())
            z.metadata["control_points"] = [[c.x() / self.template.width, c.y() / self.template.height] for c in cps]
            self.zones_changed.emit(); self.update(); return

    def mouseReleaseEvent(self, e: QMouseEvent):
        if not self.template or e.button() != Qt.LeftButton:
            return
        if self._drawing:
            self._drawing = False
            r = self._current_rect.normalized()
            if r.width() > 10 and r.height() > 10:
                z = Zone(
                    id=str(uuid.uuid4()),
                    name=f"{self.current_zone_type.value}_{len(self.template.zones)+1}",
                    zone_type=self.current_zone_type,
                    x=r.x() / self.template.width,
                    y=r.y() / self.template.height,
                    width=r.width() / self.template.width,
                    height=r.height() / self.template.height,
                    metadata={"control_points": [
                        [r.x() / self.template.width, r.y() / self.template.height],
                        [(r.x() + r.width()) / self.template.width, r.y() / self.template.height],
                        [r.x() / self.template.width, (r.y() + r.height()) / self.template.height],
                        [(r.x() + r.width()) / self.template.width, (r.y() + r.height()) / self.template.height],
                    ]},
                )
                self.template.zones.append(z)
                self.selected_zone = len(self.template.zones) - 1
                self.selection_changed.emit(self.selected_zone)
                self.zones_changed.emit()
            self._current_rect = QRect()

        self._moving = False
        self._drag_corner = -1
        self.update()

    def _hit_zone_or_control(self, p: QPoint) -> tuple[int, int]:
        if not self.template:
            return -1, -1
        for i in range(len(self.template.zones) - 1, -1, -1):
            z = self.template.zones[i]
            cps = self._control_points_abs(z)
            for ci, cp in enumerate(cps):
                if abs(cp.x() - p.x()) <= 8 and abs(cp.y() - p.y()) <= 8:
                    return i, ci
            if self._zone_rect_abs(z).contains(p):
                return i, -1
        return -1, -1

    def _hit_anchor(self, p: QPoint) -> int:
        if not self.template:
            return -1
        for i in range(len(self.template.anchors) - 1, -1, -1):
            a = self.template.anchors[i]
            ax = a.x * self.template.width
            ay = a.y * self.template.height
            if abs(ax - p.x()) <= 10 and abs(ay - p.y()) <= 10:
                return i
        return -1


    def keyPressEvent(self, e: QKeyEvent):
        if not self.template:
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected_zone < len(self.template.zones):
                del self.template.zones[self.selected_zone]
                self.selected_zone = -1
                self.selection_changed.emit(-1)
                self.zones_changed.emit()
                self.update()
                return
            if 0 <= self.selected_anchor < len(self.template.anchors):
                del self.template.anchors[self.selected_anchor]
                self.selected_anchor = -1
                self.zones_changed.emit()
                self.update()
                return

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self.pixmap:
            p.fillRect(self.rect(), Qt.lightGray); return

        p.drawPixmap(QRect(0, 0, int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom)), self.pixmap)
        if not self.template:
            return

        p.setPen(QPen(Qt.black, 2)); p.setBrush(Qt.black)
        for i, a in enumerate(self.template.anchors):
            if i == self.selected_anchor:
                p.setPen(QPen(QColor(255, 180, 0), 2)); p.setBrush(QColor(255, 180, 0))
            else:
                p.setPen(QPen(Qt.black, 2)); p.setBrush(Qt.black)
            ax = a.x * self.template.width * self.zoom
            ay = a.y * self.template.height * self.zoom
            p.drawRect(QRectF(ax - 4, ay - 4, 8, 8))
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.drawText(QPointF(ax + 6, ay - 6), str(getattr(a, "name", "") or f"A{i+1}"))

        # Detected anchors from test-recognition pass.
        if self.detected_anchor_points:
            p.setPen(QPen(QColor(40, 180, 255), 2)); p.setBrush(Qt.NoBrush)
            for ax, ay in self.detected_anchor_points:
                x = ax * self.zoom
                y = ay * self.zoom
                p.drawLine(QPointF(x - 6, y - 6), QPointF(x + 6, y + 6))
                p.drawLine(QPointF(x - 6, y + 6), QPointF(x + 6, y - 6))

        for i, z in enumerate(self.template.zones):
            zr = self._zone_rect_abs(z)
            p.setPen(QPen(QColor(220, 60, 60), 2)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(zr.x() * self.zoom, zr.y() * self.zoom, zr.width() * self.zoom, zr.height() * self.zoom))

            cps = self._control_points_abs(z)
            p.setPen(QPen(QColor(30, 120, 250), 1)); p.setBrush(QColor(30, 120, 250))
            for cp in cps:
                p.drawEllipse(QPointF(cp.x() * self.zoom, cp.y() * self.zoom), 4, 4)

            if self.preview_mode and z.grid:
                self._draw_grid_preview(p, z)
            if z.id in self.digit_zone_debug and z.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                self._draw_digit_zone_debug(p, z, self.digit_zone_debug[z.id])
            if z.id in self.recognition_overlay and z.grid:
                self._draw_recognition_overlay(p, z)

        if self._drawing:
            r = self._current_rect
            p.setPen(QPen(Qt.green, 2, Qt.DashLine)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(r.x() * self.zoom, r.y() * self.zoom, r.width() * self.zoom, r.height() * self.zoom))

    def _draw_grid_preview(self, painter: QPainter, zone: Zone):
        assert self.template and zone.grid
        rows, cols = max(1, zone.grid.rows), max(1, zone.grid.cols)
        painter.setPen(QPen(QColor(50, 120, 240), 1))  # blue template bubbles
        for idx, (bx, by) in enumerate(zone.grid.bubble_positions):
            x = bx * self.template.width * self.zoom
            y = by * self.template.height * self.zoom
            painter.drawEllipse(QPointF(x, y), 2.5, 2.5)

            r = idx // cols
            c = idx % cols

            if zone.zone_type == ZoneType.MCQ_BLOCK:
                if c == 0:
                    painter.drawText(QPointF(x - 22, y + 4), str(zone.grid.question_start + r))
                if c < len(zone.grid.options):
                    painter.drawText(QPointF(x + 4, y - 2), zone.grid.options[c])

            elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
                spq = int(zone.metadata.get("statements_per_question", 4))
                qpb = int(zone.metadata.get("questions_per_block", 2))
                q_no = zone.grid.question_start + (r // max(1, spq))
                stmt_labels = [chr(ord("a") + i) for i in range(spq)]
                stmt = stmt_labels[r % max(1, spq)]
                if c == 0:
                    painter.drawText(QPointF(x - 30, y + 4), f"{q_no}{stmt}")
                if c < len(zone.grid.options):
                    painter.drawText(QPointF(x + 4, y - 2), zone.grid.options[c])

            elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
                digits = int(zone.metadata.get("digits_per_answer", 5))
                q_no = zone.grid.question_start + (c // max(1, digits))
                d_no = (c % max(1, digits)) + 1
                if r == 0:
                    painter.drawText(QPointF(x - 12, y - 6), f"Q{q_no}D{d_no}")
                if r < len(zone.grid.options):
                    painter.drawText(QPointF(x + 4, y - 2), zone.grid.options[r])

            elif zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                if r == 0:
                    painter.drawText(QPointF(x - 10, y - 6), f"C{c+1}")
                if r < len(zone.grid.options):
                    painter.drawText(QPointF(x + 4, y - 2), zone.grid.options[r])

    def _draw_recognition_overlay(self, painter: QPainter, zone: Zone):
        assert self.template and zone.grid
        states = self.recognition_overlay[zone.id]
        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            x = bx * self.template.width * self.zoom
            y = by * self.template.height * self.zoom
            seen = i < len(states) and states[i]
            if seen:
                painter.setPen(QPen(QColor(0, 200, 0), 2)); painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(x - 5, y - 5), QPointF(x + 5, y + 5))
                painter.drawLine(QPointF(x - 5, y + 5), QPointF(x + 5, y - 5))
            else:
                painter.setPen(QPen(QColor(230, 60, 60), 1)); painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(x, y), 3.5, 3.5)

    def _draw_digit_zone_debug(self, painter: QPainter, zone: Zone, debug: dict[str, object]) -> None:
        if not self.template:
            return
        zr = self._zone_rect_abs(zone)
        left = zr.x() * self.zoom
        right = (zr.x() + zr.width()) * self.zoom
        top = zr.y() * self.zoom
        bottom = (zr.y() + zr.height()) * self.zoom

        model = (self.template.metadata or {}).get("digit_model", {}) if self.template else {}
        show_grid = bool(model.get("show_grid", True))
        show_anchors = bool(model.get("show_anchors", True))
        show_points = bool(model.get("show_points", True))

        painter.setBrush(Qt.NoBrush)
        if show_grid:
            painter.setPen(QPen(QColor(255, 140, 0), 1.5))
            for seg in debug.get("row_segments", []) or []:
                start, end = seg
                painter.drawLine(QPointF(float(start[0]) * self.zoom, float(start[1]) * self.zoom), QPointF(float(end[0]) * self.zoom, float(end[1]) * self.zoom))
            for y in debug.get("row_lines", []) or []:
                yy = float(y) * self.zoom
                painter.drawLine(QPointF(left, yy), QPointF(right, yy))

            painter.setPen(QPen(QColor(0, 180, 0), 1.2))
            segments = debug.get("col_segments", []) or []
            if segments:
                for start, end in segments:
                    painter.drawLine(
                        QPointF(float(start[0]) * self.zoom, float(start[1]) * self.zoom),
                        QPointF(float(end[0]) * self.zoom, float(end[1]) * self.zoom),
                    )
            else:
                for x in debug.get("col_lines", []) or []:
                    xx = float(x) * self.zoom
                    painter.drawLine(QPointF(xx, top), QPointF(xx, bottom))

        if show_anchors:
            painter.setPen(QPen(QColor(40, 120, 255), 2))
            for x, y in debug.get("anchor_points", []) or []:
                painter.drawEllipse(QPointF(float(x) * self.zoom, float(y) * self.zoom), 4, 4)
            anchor_line = debug.get("anchor_line", []) or []
            if len(anchor_line) == 2:
                painter.drawLine(QPointF(float(anchor_line[0][0]) * self.zoom, float(anchor_line[0][1]) * self.zoom), QPointF(float(anchor_line[1][0]) * self.zoom, float(anchor_line[1][1]) * self.zoom))

        painter.setPen(QPen(QColor(0, 200, 200), 1.2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        for x, y, w, h in debug.get("guide_regions", []) or []:
            painter.drawRect(QRectF(float(x) * self.zoom, float(y) * self.zoom, float(w) * self.zoom, float(h) * self.zoom))

        if show_points:
            painter.setPen(QPen(QColor(220, 40, 40), 2))
            for x, y in debug.get("bubble_centers", []) or []:
                painter.drawEllipse(QPointF(float(x) * self.zoom, float(y) * self.zoom), 2.5, 2.5)
            painter.setPen(QPen(QColor(0, 220, 220), 2))
            for x, y in debug.get("guide_points", []) or []:
                px = float(x) * self.zoom
                py = float(y) * self.zoom
                painter.drawLine(QPointF(px - 6, py - 6), QPointF(px + 6, py + 6))
                painter.drawLine(QPointF(px - 6, py + 6), QPointF(px + 6, py - 6))
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            for x, y in debug.get("recognized_points", []) or []:
                px = float(x) * self.zoom
                py = float(y) * self.zoom
                painter.drawLine(QPointF(px - 5, py - 5), QPointF(px + 5, py + 5))
                painter.drawLine(QPointF(px - 5, py + 5), QPointF(px + 5, py - 5))


class TemplateEditorWindow(QMainWindow):
    def __init__(self, parent=None, on_template_saved=None):
        super().__init__(parent)
        self.setWindowTitle("Template Editor")
        screen = self.screen().availableGeometry() if self.screen() is not None else QRect(0, 0, 1620, 940)
        self.setGeometry(screen)
        self.setMinimumSize(max(1200, int(screen.width() * 0.8)), max(800, int(screen.height() * 0.8)))
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.template_engine = TemplateEngine()
        self.omr = OMRProcessor()
        self.template: Template | None = None
        self.clipboard_zone: Zone | None = None
        self.preview_ok = False
        self.test_ok = False
        self._sync = False
        self.template_file_path: str | None = None
        self._original_pixmap: QPixmap | None = None
        self.template_dirty = False
        self.on_template_saved = on_template_saved

        self.canvas = TemplateCanvas()
        self.canvas.selection_changed.connect(self._load_props)
        self.canvas.zones_changed.connect(self._on_zone_changed)

        scroll = QScrollArea(); scroll.setWidgetResizable(False); scroll.setWidget(self.canvas)

        # Grouped actions (menu style like office apps)
        self.act_load_blank = QAction("Load Blank Paper", self); self.act_load_blank.triggered.connect(self.load_image)
        self.act_open_template = QAction("Open Template", self); self.act_open_template.triggered.connect(self.open_template)
        self.act_save = QAction("Save", self); self.act_save.triggered.connect(self.save_template)
        self.act_save_as = QAction("Save As", self); self.act_save_as.triggered.connect(self.save_template_as)

        self.act_preview = QAction("Preview", self); self.act_preview.triggered.connect(self.preview_template)
        self.act_test_recognition = QAction("Test Recognition", self); self.act_test_recognition.triggered.connect(self.test_recognition)
        self.act_template_qc = QAction("Template QC", self); self.act_template_qc.triggered.connect(self.run_template_quality_check)

        self.act_copy = QAction("Copy Block", self); self.act_copy.triggered.connect(self.copy_block)
        self.act_paste = QAction("Paste Block", self); self.act_paste.triggered.connect(self.paste_block)
        self.act_duplicate = QAction("Duplicate Block", self); self.act_duplicate.triggered.connect(self.duplicate_block)
        self.act_delete = QAction("Delete Block", self); self.act_delete.triggered.connect(self.delete_selected_block)
        self.act_delete_anchor = QAction("Delete Anchor", self); self.act_delete_anchor.triggered.connect(self.delete_selected_anchor)
        self.act_generate_digit_anchors = QAction("Generate Digit Anchors", self); self.act_generate_digit_anchors.triggered.connect(self.generate_digit_zone_anchors)
        self.act_snap_grid = QAction("Snap Grid", self); self.act_snap_grid.triggered.connect(self.snap_grid_to_detected_bubbles)
        self.act_zoom_in = QAction("Zoom +", self); self.act_zoom_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15))
        self.act_zoom_out = QAction("Zoom -", self); self.act_zoom_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15))

        self._build_menus()
        self._assign_action_icons()

        self.template_toolbar = QToolBar("Template")
        self.template_toolbar.setMovable(False)
        self.template_toolbar.setFloatable(False)
        self.template_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.template_toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(self.template_toolbar)
        for act in [
            self.act_load_blank,
            self.act_open_template,
            self.act_save,
            self.act_save_as,
            self.act_preview,
            self.act_test_recognition,
            self.act_template_qc,
            self.act_copy,
            self.act_paste,
            self.act_duplicate,
            self.act_delete,
            self.act_delete_anchor,
            self.act_generate_digit_anchors,
            self.act_snap_grid,
        ]:
            self.template_toolbar.addAction(act)

        self.anchor_btn = QPushButton("Add Anchor"); self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(self._toggle_add_anchor_mode)
        self.template_toolbar.addWidget(self.anchor_btn)
        self.digit_anchor_btn = QPushButton("Add Digit Anchor"); self.digit_anchor_btn.setCheckable(True)
        self.digit_anchor_btn.toggled.connect(self._toggle_add_digit_anchor_mode)
        self.template_toolbar.addWidget(self.digit_anchor_btn)

        self.template_toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_type = QComboBox(); self.zone_type.addItems([z.value for z in [ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK, ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK]])
        self.zone_type.currentTextChanged.connect(self._on_zone_type_changed)
        self.template_toolbar.addWidget(self.zone_type)

        self.template_toolbar.addWidget(QLabel(" Alignment: "))
        self.align_profile_combo = QComboBox()
        self.align_profile_combo.addItem("Auto", "auto")
        self.align_profile_combo.addItem("Legacy", "legacy")
        self.align_profile_combo.addItem("Border", "border")
        self.align_profile_combo.addItem("One-side ruler", "one_side")
        self.align_profile_combo.addItem("Hybrid", "hybrid")
        self.align_profile_combo.currentIndexChanged.connect(self._on_alignment_profile_changed)
        self.template_toolbar.addWidget(self.align_profile_combo)

        self.template_toolbar.addWidget(QLabel(" Fill threshold: "))
        self.fill_threshold_spin = QDoubleSpinBox()
        self.fill_threshold_spin.setDecimals(2)
        self.fill_threshold_spin.setRange(0.05, 0.95)
        self.fill_threshold_spin.setSingleStep(0.01)
        self.fill_threshold_spin.setValue(0.45)
        self.fill_threshold_spin.setToolTip("Tăng để giảm nhận nhầm vết mờ/vết tẩy; giảm nếu bỏ sót nét tô nhẹ.")
        self.fill_threshold_spin.valueChanged.connect(self._on_fill_threshold_changed)
        self.template_toolbar.addWidget(self.fill_threshold_spin)

        self.template_toolbar.addAction(self.act_zoom_in)
        self.template_toolbar.addAction(self.act_zoom_out)

        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMinimumHeight(140)
        self.prop_panel = self._build_prop_panel()

        image_results_splitter = QSplitter(Qt.Vertical)
        image_results_splitter.addWidget(scroll)
        image_results_splitter.addWidget(self.result_box)
        image_results_splitter.setStretchFactor(0, 5)
        image_results_splitter.setStretchFactor(1, 1)
        image_results_splitter.setSizes([760, 180])

        center = QWidget(); layout = QHBoxLayout(center)
        layout.addWidget(image_results_splitter, 1); layout.addWidget(self.prop_panel)
        self.setCentralWidget(center)

    def _assign_action_icons(self) -> None:
        s = self.style()
        self.act_load_blank.setIcon(s.standardIcon(QStyle.SP_DirOpenIcon))
        self.act_open_template.setIcon(s.standardIcon(QStyle.SP_FileIcon))
        self.act_save.setIcon(s.standardIcon(QStyle.SP_DialogSaveButton))
        self.act_save_as.setIcon(s.standardIcon(QStyle.SP_DriveFDIcon))
        self.act_preview.setIcon(s.standardIcon(QStyle.SP_FileDialogContentsView))
        self.act_test_recognition.setIcon(s.standardIcon(QStyle.SP_MediaPlay))
        self.act_template_qc.setIcon(s.standardIcon(QStyle.SP_MessageBoxInformation))
        self.act_copy.setIcon(s.standardIcon(QStyle.SP_FileDialogDetailedView))
        self.act_paste.setIcon(s.standardIcon(QStyle.SP_DialogResetButton))
        self.act_duplicate.setIcon(s.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_delete.setIcon(s.standardIcon(QStyle.SP_TrashIcon))
        self.act_delete_anchor.setIcon(s.standardIcon(QStyle.SP_BrowserStop))
        self.act_generate_digit_anchors.setIcon(s.standardIcon(QStyle.SP_FileDialogListView))
        self.act_snap_grid.setIcon(s.standardIcon(QStyle.SP_BrowserReload))
        self.act_zoom_in.setIcon(s.standardIcon(QStyle.SP_ArrowUp))
        self.act_zoom_out.setIcon(s.standardIcon(QStyle.SP_ArrowDown))

    def _build_menus(self) -> None:
        menu = self.menuBar()

        m_file = menu.addMenu("File")
        m_file.addAction(self.act_load_blank)
        m_file.addAction(self.act_open_template)
        m_file.addSeparator()
        m_file.addAction(self.act_save)
        m_file.addAction(self.act_save_as)

        m_edit = menu.addMenu("Edit")
        m_edit.addAction(self.act_copy)
        m_edit.addAction(self.act_paste)
        m_edit.addAction(self.act_duplicate)
        m_edit.addAction(self.act_delete)
        m_edit.addAction(self.act_delete_anchor)
        m_edit.addAction(self.act_generate_digit_anchors)

        m_recog = menu.addMenu("Recognition")
        m_recog.addAction(self.act_preview)
        m_recog.addAction(self.act_test_recognition)
        m_recog.addAction(self.act_template_qc)
        m_recog.addAction(self.act_snap_grid)

        m_view = menu.addMenu("View")
        m_view.addAction(self.act_zoom_in)
        m_view.addAction(self.act_zoom_out)

    def _build_prop_panel(self) -> QWidget:
        w = QWidget(); w.setFixedWidth(360)
        l = QVBoxLayout(w); l.addWidget(QLabel("Semantic Grid Properties"))
        f = QFormLayout()
        self.p_qstart = QSpinBox(); self.p_qstart.setRange(1, 10000); self.p_qstart.setValue(1)
        self.p_total = QSpinBox(); self.p_total.setRange(1, 2000); self.p_total.setValue(10)
        self.p_choices = QSpinBox(); self.p_choices.setRange(2, 20); self.p_choices.setValue(4)
        self.p_qpc = QSpinBox(); self.p_qpc.setRange(1, 500); self.p_qpc.setValue(10)
        self.p_cols = QSpinBox(); self.p_cols.setRange(1, 50); self.p_cols.setValue(1)
        self.p_scale = QDoubleSpinBox(); self.p_scale.setRange(0.1, 3.0); self.p_scale.setSingleStep(0.05); self.p_scale.setValue(1.0)
        self.p_ox = QDoubleSpinBox(); self.p_ox.setRange(-0.5, 0.5); self.p_ox.setSingleStep(0.01)
        self.p_oy = QDoubleSpinBox(); self.p_oy.setRange(-0.5, 0.5); self.p_oy.setSingleStep(0.01)
        self.p_qpb = QSpinBox(); self.p_qpb.setRange(1, 20); self.p_qpb.setValue(2)
        self.p_spq = QSpinBox(); self.p_spq.setRange(1, 10); self.p_spq.setValue(4)
        self.p_digits = QSpinBox(); self.p_digits.setRange(1, 20); self.p_digits.setValue(5)
        self.p_cps = QSpinBox(); self.p_cps.setRange(2, 5); self.p_cps.setValue(2)
        self.p_rows = QSpinBox(); self.p_rows.setRange(2, 20); self.p_rows.setValue(10)
        self.p_columns = QSpinBox(); self.p_columns.setRange(1, 20); self.p_columns.setValue(8)
        self.p_digit_map = QLineEdit(); self.p_digit_map.setText("0,1,2,3,4,5,6,7,8,9")
        self.p_sign_row = QSpinBox(); self.p_sign_row.setRange(1, 50); self.p_sign_row.setValue(1)
        self.p_decimal_row = QSpinBox(); self.p_decimal_row.setRange(1, 50); self.p_decimal_row.setValue(2)
        self.p_digit_start_row = QSpinBox(); self.p_digit_start_row.setRange(1, 50); self.p_digit_start_row.setValue(3)
        self.p_sign_columns = QLineEdit(); self.p_sign_columns.setText("1")
        self.p_decimal_columns = QLineEdit(); self.p_decimal_columns.setText("2,3")
        self.p_sign_symbol = QLineEdit(); self.p_sign_symbol.setText("-")
        self.p_decimal_symbol = QLineEdit(); self.p_decimal_symbol.setText(".")
        self.p_digit_col_scale = QDoubleSpinBox(); self.p_digit_col_scale.setRange(0.2, 3.0); self.p_digit_col_scale.setSingleStep(0.05); self.p_digit_col_scale.setValue(1.0)
        self.p_digit_row_scale = QDoubleSpinBox(); self.p_digit_row_scale.setRange(0.2, 3.0); self.p_digit_row_scale.setSingleStep(0.05); self.p_digit_row_scale.setValue(1.0)
        self.p_digit_rotation = QDoubleSpinBox(); self.p_digit_rotation.setRange(-45.0, 45.0); self.p_digit_rotation.setSingleStep(0.5); self.p_digit_rotation.setValue(0.0)
        self.p_sid_offset_x = QDoubleSpinBox(); self.p_sid_offset_x.setRange(-2000.0, 2000.0); self.p_sid_offset_x.setSingleStep(1.0)
        self.p_sid_offset_y = QDoubleSpinBox(); self.p_sid_offset_y.setRange(-2000.0, 2000.0); self.p_sid_offset_y.setSingleStep(1.0)
        self.p_exam_offset_x = QDoubleSpinBox(); self.p_exam_offset_x.setRange(-2000.0, 2000.0); self.p_exam_offset_x.setSingleStep(1.0)
        self.p_exam_offset_y = QDoubleSpinBox(); self.p_exam_offset_y.setRange(-2000.0, 2000.0); self.p_exam_offset_y.setSingleStep(1.0)
        self.p_digit_center_ratio = QDoubleSpinBox(); self.p_digit_center_ratio.setRange(0.10, 0.90); self.p_digit_center_ratio.setSingleStep(0.01); self.p_digit_center_ratio.setValue(0.35)
        self.show_digit_grid_chk = QCheckBox("Show Grid"); self.show_digit_grid_chk.setChecked(True)
        self.show_digit_anchors_chk = QCheckBox("Show Anchors"); self.show_digit_anchors_chk.setChecked(True)
        self.show_digit_points_chk = QCheckBox("Show Sampling Points"); self.show_digit_points_chk.setChecked(True)

        self._prop_controls = [
            ("question_start", self.p_qstart), ("total_questions", self.p_total), ("choices_per_question", self.p_choices),
            ("questions_per_column", self.p_qpc), ("column_count", self.p_cols), ("grid_scale", self.p_scale),
            ("offset_x", self.p_ox), ("offset_y", self.p_oy), ("questions_per_block", self.p_qpb),
            ("statements_per_question", self.p_spq), ("choices_per_statement", self.p_cps), ("digits_per_answer", self.p_digits), ("rows", self.p_rows), ("columns", self.p_columns), ("digit_map", self.p_digit_map),
            ("sign_row", self.p_sign_row), ("decimal_row", self.p_decimal_row), ("digit_start_row", self.p_digit_start_row),
            ("sign_columns", self.p_sign_columns), ("decimal_columns", self.p_decimal_columns), ("sign_symbol", self.p_sign_symbol), ("decimal_symbol", self.p_decimal_symbol),
            ("digit_col_spacing_scale", self.p_digit_col_scale), ("digit_row_spacing_scale", self.p_digit_row_scale), ("digit_rotation_deg", self.p_digit_rotation),
            ("sid_offset_x", self.p_sid_offset_x), ("sid_offset_y", self.p_sid_offset_y), ("exam_offset_x", self.p_exam_offset_x), ("exam_offset_y", self.p_exam_offset_y), ("digit_center_ratio", self.p_digit_center_ratio),
            ("show_digit_grid", self.show_digit_grid_chk), ("show_digit_anchors", self.show_digit_anchors_chk), ("show_digit_points", self.show_digit_points_chk),
        ]
        self._prop_rows = {}
        for name, widget in self._prop_controls:
            lbl = QLabel(name)
            f.addRow(lbl, widget)
            self._prop_rows[name] = (lbl, widget)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._prop_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._prop_changed)
            else:
                widget.textChanged.connect(self._prop_changed)
        l.addLayout(f)
        self.btn_regen = QPushButton("Regenerate Grid"); self.btn_regen.clicked.connect(self.regenerate_selected_grid)
        l.addWidget(self.btn_regen); l.addStretch(1)
        return w


    def _apply_block_property_visibility(self, zone_type: ZoneType | None) -> None:
        visible = {
            "question_start": True,
            "grid_scale": True,
            "offset_x": True,
            "offset_y": True,
        }
        if zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            visible.update({
                "rows": True, "columns": True, "digit_map": True,
                "digit_col_spacing_scale": True, "digit_row_spacing_scale": True, "digit_rotation_deg": True,
                "sid_offset_x": True, "sid_offset_y": True, "exam_offset_x": True, "exam_offset_y": True,
                "show_digit_grid": True, "show_digit_anchors": True, "show_digit_points": True,
            })
            visible["grid_scale"] = False
            visible["offset_x"] = False
            visible["offset_y"] = False
        elif zone_type == ZoneType.MCQ_BLOCK:
            visible.update({"questions_per_block": True, "choices_per_question": True})
        elif zone_type == ZoneType.TRUE_FALSE_BLOCK:
            visible.update({"questions_per_block": True, "statements_per_question": True, "choices_per_statement": True})
        elif zone_type == ZoneType.NUMERIC_BLOCK:
            visible.update({
                "questions_per_block": True,
                "digits_per_answer": True,
                "rows": True,
                "digit_map": True,
                "sign_row": True,
                "decimal_row": True,
                "digit_start_row": True,
                "sign_columns": True,
                "decimal_columns": True,
                "sign_symbol": True,
                "decimal_symbol": True,
            })
        elif zone_type == ZoneType.STUDENT_ID_BLOCK:
            visible.update({"rows": True, "columns": True, "digit_map": True})
        elif zone_type == ZoneType.EXAM_CODE_BLOCK:
            visible.update({"rows": True, "columns": True, "digit_map": True})

        for name, (lbl, widget) in self._prop_rows.items():
            show = visible.get(name, False)
            lbl.setVisible(show)
            widget.setVisible(show)

    def _selected_zone(self) -> Zone | None:
        if not self.template: return None
        i = self.canvas.selected_zone
        if i < 0 or i >= len(self.template.zones): return None
        return self.template.zones[i]

    def _on_zone_changed(self):
        self._mark_dirty()
        if self.canvas.preview_mode:
            self.regenerate_selected_grid(auto=True)
            self.preview_template()

    def _on_zone_type_changed(self, text: str):
        zt = ZoneType(text)
        self.canvas.current_zone_type = zt
        z = self._selected_zone()
        if not z or z.zone_type == zt:
            return

        z.zone_type = zt
        self._mark_dirty()
        if zt == ZoneType.STUDENT_ID_BLOCK:
            z.metadata.setdefault("rows", 10)
            z.metadata.setdefault("columns", 8)
            z.metadata.setdefault("digit_map", list(range(10)))
        elif zt == ZoneType.EXAM_CODE_BLOCK:
            z.metadata.setdefault("rows", 10)
            z.metadata.setdefault("columns", 4)
            z.metadata.setdefault("digit_map", list(range(10)))
        self.regenerate_selected_grid(auto=True)
        self._load_props(self.canvas.selected_zone)
        self.canvas.update()

    def _load_props(self, _idx: int):
        z = self._selected_zone()
        enabled = bool(z and z.zone_type in BLOCK_TYPES)
        for w in [self.p_qstart, self.p_total, self.p_choices, self.p_qpc, self.p_cols, self.p_scale, self.p_ox, self.p_oy, self.p_qpb, self.p_spq, self.p_cps, self.p_digits, self.p_rows, self.p_columns, self.p_digit_map, self.p_sign_row, self.p_decimal_row, self.p_digit_start_row, self.p_sign_columns, self.p_decimal_columns, self.p_sign_symbol, self.p_decimal_symbol, self.btn_regen]:
            w.setEnabled(enabled)
        self._apply_block_property_visibility(z.zone_type if z else None)
        if not enabled:
            return
        md = z.metadata
        self._sync = True
        self.zone_type.setCurrentText(z.zone_type.value)
        self.p_qstart.setValue(int(md.get("question_start", 1)))
        self.p_total.setValue(int(md.get("total_questions", md.get("questions_per_block", 10))))
        self.p_choices.setValue(int(md.get("choices_per_question", 4)))
        self.p_qpc.setValue(int(md.get("questions_per_column", 10)))
        self.p_cols.setValue(int(md.get("column_count", 1)))
        self.p_scale.setValue(float(md.get("grid_scale", 1.0)))
        self.p_ox.setValue(float(md.get("bubble_offset_x", 0.0)))
        self.p_oy.setValue(float(md.get("bubble_offset_y", 0.0)))
        self.p_qpb.setValue(int(md.get("questions_per_block", md.get("total_questions", 10))))
        self.p_spq.setValue(int(md.get("statements_per_question", 4)))
        self.p_cps.setValue(int(md.get("choices_per_statement", 2)))
        self.p_digits.setValue(int(md.get("digits_per_answer", 5)))
        default_rows = 10
        default_cols = 8 if z.zone_type == ZoneType.STUDENT_ID_BLOCK else 4 if z.zone_type == ZoneType.EXAM_CODE_BLOCK else 8
        self.p_rows.setValue(int(md.get("rows", default_rows)))
        self.p_columns.setValue(int(md.get("columns", default_cols)))
        default_digit_map = list(range(10)) if z.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else list(range(self.p_rows.value()))
        self.p_digit_map.setText(",".join(str(x) for x in md.get("digit_map", default_digit_map)))
        self.p_sign_row.setValue(int(md.get("sign_row", 1)))
        self.p_decimal_row.setValue(int(md.get("decimal_row", 2)))
        self.p_digit_start_row.setValue(int(md.get("digit_start_row", 3)))
        self.p_sign_columns.setText(",".join(str(x) for x in md.get("sign_columns", [1])))
        self.p_decimal_columns.setText(",".join(str(x) for x in md.get("decimal_columns", [2, 3])))
        self.p_sign_symbol.setText(str(md.get("sign_symbol", "-")))
        self.p_decimal_symbol.setText(str(md.get("decimal_symbol", ".")))
        dmodel = self._digit_model()
        if z.zone_type == ZoneType.STUDENT_ID_BLOCK:
            self.p_digit_col_scale.setValue(float(dmodel.get("sid_col_spacing_scale", 1.0) or 1.0))
            self.p_digit_row_scale.setValue(float(dmodel.get("sid_row_spacing_scale", 1.0) or 1.0))
        else:
            self.p_digit_col_scale.setValue(float(dmodel.get("exam_col_spacing_scale", 1.0) or 1.0))
            self.p_digit_row_scale.setValue(float(dmodel.get("exam_row_spacing_scale", 1.0) or 1.0))
        self.p_digit_rotation.setValue(float(dmodel.get("rotation_deg", 0.0) or 0.0))
        sid_off = dmodel.get("offset_sid", [0.0, 0.0]) or [0.0, 0.0]
        exam_off = dmodel.get("offset_exam", [0.0, 0.0]) or [0.0, 0.0]
        self.p_sid_offset_x.setValue(float(sid_off[0] if len(sid_off) > 0 else 0.0))
        self.p_sid_offset_y.setValue(float(sid_off[1] if len(sid_off) > 1 else 0.0))
        self.p_exam_offset_x.setValue(float(exam_off[0] if len(exam_off) > 0 else 0.0))
        self.p_exam_offset_y.setValue(float(exam_off[1] if len(exam_off) > 1 else 0.0))
        self.show_digit_grid_chk.setChecked(bool(dmodel.get("show_grid", True)))
        self.show_digit_anchors_chk.setChecked(bool(dmodel.get("show_anchors", True)))
        self.show_digit_points_chk.setChecked(bool(dmodel.get("show_points", True)))
        if z.zone_type == ZoneType.STUDENT_ID_BLOCK:
            self.p_digit_center_ratio.setValue(float(dmodel.get("sid_center_ratio", 0.35) or 0.35))
        else:
            self.p_digit_center_ratio.setValue(float(dmodel.get("exam_center_ratio", 0.35) or 0.35))
        self._sync = False

    def _prop_changed(self, *_args):
        if self._sync:
            return
        if self.template:
            model = self._digit_model()
            model["rotation_deg"] = float(self.p_digit_rotation.value())
            model["offset_sid"] = [float(self.p_sid_offset_x.value()), float(self.p_sid_offset_y.value())]
            model["offset_exam"] = [float(self.p_exam_offset_x.value()), float(self.p_exam_offset_y.value())]
            model["show_grid"] = bool(self.show_digit_grid_chk.isChecked())
            model["show_anchors"] = bool(self.show_digit_anchors_chk.isChecked())
            model["show_points"] = bool(self.show_digit_points_chk.isChecked())
            z = self._selected_zone()
            if z and z.zone_type == ZoneType.STUDENT_ID_BLOCK:
                model["sid_col_spacing_scale"] = float(self.p_digit_col_scale.value())
                model["sid_row_spacing_scale"] = float(self.p_digit_row_scale.value())
                model["sid_center_ratio"] = float(self.p_digit_center_ratio.value())
            elif z and z.zone_type == ZoneType.EXAM_CODE_BLOCK:
                model["exam_col_spacing_scale"] = float(self.p_digit_col_scale.value())
                model["exam_row_spacing_scale"] = float(self.p_digit_row_scale.value())
                model["exam_center_ratio"] = float(self.p_digit_center_ratio.value())
            self.template.metadata["digit_model"] = model
        self.regenerate_selected_grid(auto=True)
        self._refresh_digit_model_overlay()

    def _digit_model(self) -> dict:
        if not self.template:
            return {}
        model = dict((self.template.metadata or {}).get("digit_model", {}) or {})
        model.setdefault("sid_col_spacing_scale", 1.0)
        model.setdefault("sid_row_spacing_scale", 1.0)
        model.setdefault("exam_col_spacing_scale", 1.0)
        model.setdefault("exam_row_spacing_scale", 1.0)
        model.setdefault("rotation_deg", 0.0)
        model.setdefault("offset_sid", [0.0, 0.0])
        model.setdefault("offset_exam", [0.0, 0.0])
        model.setdefault("show_grid", True)
        model.setdefault("show_anchors", True)
        model.setdefault("show_points", True)
        model.setdefault("sid_center_ratio", 0.35)
        model.setdefault("exam_center_ratio", 0.35)
        return model

    def _manual_digit_anchor_points(self) -> list[tuple[float, float]]:
        if not self.template:
            return []
        indexed: list[tuple[int, tuple[float, float]]] = []
        for a in (self.template.anchors or []):
            name = str(getattr(a, 'name', '') or '')
            if not name.startswith('DIGIT_ANCHOR_'):
                continue
            try:
                idx = int(name.rsplit('_', 1)[-1])
            except Exception:
                idx = len(indexed) + 1
            indexed.append((idx, (float(a.x * self.template.width if a.x <= 1.0 else a.x), float(a.y * self.template.height if a.y <= 1.0 else a.y))))
        return [pt for _, pt in sorted(indexed, key=lambda item: item[0])]

    def _generate_digit_block_from_model(self, zone: Zone) -> dict[str, object] | None:
        if not self.template or zone.zone_type not in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            return None
        anchors = self._manual_digit_anchor_points()
        if len(anchors) < 2:
            return None
        rows = max(1, int(zone.metadata.get('rows', 10)))
        cols = max(1, int(zone.metadata.get('columns', 8 if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else 4)))
        model = self._digit_model()
        ref_anchor = np.array(anchors[0], dtype=np.float32)
        top_points = np.array(anchors[1:] if len(anchors) > 2 else anchors, dtype=np.float32)
        if len(top_points) == 0:
            return None
        # Need 11 digit anchors total for 10 rows: anchor #1 handwritten separator, anchors #2..#11 ruler marks.
        required_anchor_count = rows + 1
        if len(top_points) > required_anchor_count:
            top_points = top_points[:required_anchor_count]
        elif len(top_points) < required_anchor_count and len(top_points) >= 2:
            interp = []
            for idx in range(required_anchor_count):
                alpha = 0.0 if required_anchor_count <= 1 else float(idx / max(1, required_anchor_count - 1))
                interp.append(((1.0 - alpha) * top_points[0]) + (alpha * top_points[-1]))
            top_points = np.array(interp, dtype=np.float32)
        # A2/A3 define the real ruler edge direction; digit anchors should follow this tilt.
        edge_pts = []
        for a in (self.template.anchors or []):
            name = str(getattr(a, 'name', '') or '')
            if name in {'A2', 'A3'}:
                edge_pts.append(np.array([float(a.x * self.template.width if a.x <= 1.0 else a.x), float(a.y * self.template.height if a.y <= 1.0 else a.y)], dtype=np.float32))
        if len(edge_pts) >= 2:
            edge_pts = sorted(edge_pts, key=lambda pt: float(pt[1]))
            anchor_vec = edge_pts[-1] - edge_pts[0]
        else:
            anchor_vec = top_points[-1] - top_points[0]
        norm = float(np.linalg.norm(anchor_vec))
        if norm < 1e-6:
            return None
        row_unit = anchor_vec / norm
        ang = np.deg2rad(float(model.get('rotation_deg', 0.0) or 0.0))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
        row_unit = rot @ row_unit
        col_unit = np.array([-row_unit[1], row_unit[0]], dtype=np.float32)
        zw = float(zone.width * self.template.width if zone.width <= 1.0 else zone.width)
        zh = float(zone.height * self.template.height if zone.height <= 1.0 else zone.height)
        base_col_spacing = (zw / max(1, cols - 1)) if cols > 1 else max(1.0, zw)
        base_row_spacing = (zh / max(1, rows)) if rows > 0 else max(1.0, zh)
        if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
            col_spacing = base_col_spacing * float(model.get('sid_col_spacing_scale', 1.0) or 1.0)
            row_spacing_scale = float(model.get('sid_row_spacing_scale', 1.0) or 1.0)
            row_spacing = base_row_spacing * row_spacing_scale
        else:
            col_spacing = base_col_spacing * float(model.get('exam_col_spacing_scale', 1.0) or 1.0)
            row_spacing_scale = float(model.get('exam_row_spacing_scale', 1.0) or 1.0)
            row_spacing = base_row_spacing * row_spacing_scale
        off_key = 'offset_sid' if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else 'offset_exam'
        off = model.get(off_key, [0.0, 0.0]) or [0.0, 0.0]
        ruler_anchors = [np.array(pt, dtype=np.float32) for pt in top_points]
        median_gap = row_spacing
        if len(ruler_anchors) >= 2:
            gaps = [float(np.dot(ruler_anchors[i + 1] - ruler_anchors[i], row_unit)) for i in range(len(ruler_anchors) - 1)]
            valid = [g for g in gaps if g > 1e-3]
            if valid:
                median_gap = float(np.median(valid))
        row_centers = []
        row_segments = []
        # Normalized digit ruler:
        # - anchor index 0 is the handwritten top separator / top clamp and is not used for row sampling
        # - anchors 1..N provide the real ruler marks used to derive row midpoints from consecutive pairs
        usable_anchors = ruler_anchors[:required_anchor_count]
        pair_anchors = usable_anchors

        base_row_midpoints: list[np.ndarray] = []
        if len(pair_anchors) >= 2:
            base_row_midpoints = [((pair_anchors[i] + pair_anchors[i + 1]) * 0.5) for i in range(len(pair_anchors) - 1)]
            if len(base_row_midpoints) < rows:
                last_gap_vec = pair_anchors[-1] - pair_anchors[-2]
                base_row_midpoints.append(pair_anchors[-1] + (0.5 * last_gap_vec))
        elif len(pair_anchors) == 1:
            base_row_midpoints = [pair_anchors[0] + (0.5 * median_gap * row_unit)]
        row_origin = base_row_midpoints[0] if base_row_midpoints else ref_anchor

        for r in range(rows):
            if r < len(base_row_midpoints):
                center_base = base_row_midpoints[r].copy()
            elif base_row_midpoints:
                center_base = row_origin + ((r) * median_gap * row_unit)
            else:
                center_base = ref_anchor + ((r + 0.5) * median_gap * row_unit)

            delta = center_base - row_origin
            along = float(np.dot(delta, row_unit))
            perp = delta - (along * row_unit)
            scaled_base = row_origin + perp + ((along * row_spacing_scale) * row_unit)

            center = scaled_base + (float(off[0]) * col_unit) + (float(off[1]) * row_unit)
            row_centers.append(center)

            line_center = scaled_base + (float(off[1]) * row_unit)
            row_start = line_center.copy()
            row_end = row_start - (zw * col_unit)
            row_segments.append((tuple(row_start.tolist()), tuple(row_end.tolist())))
        points=[]
        for r in range(rows):
            for c in range(cols):
                pt = row_centers[r] + (c * col_spacing * col_unit)
                points.append((float(pt[0]) / self.template.width, float(pt[1]) / self.template.height))
        col_lines=[]
        for c in range(cols):
            start = row_centers[0] + (c * col_spacing * col_unit)
            end = row_centers[-1] + (c * col_spacing * col_unit)
            col_lines.append((tuple(start.tolist()), tuple(end.tolist())))
        return {
            'bubble_positions': points,
            'anchor_points': anchors,
            'anchor_line': [tuple(ref_anchor.tolist()), tuple((ref_anchor + (row_unit * max(median_gap * rows, 1.0))).tolist())],
            'col_lines': col_lines,
            'row_segments': row_segments,
            'bubble_centers': [(p[0]*self.template.width, p[1]*self.template.height) for p in points],
        }

    def _refresh_digit_model_overlay(self) -> None:
        if not self.template:
            return
        dbg = dict(getattr(self.canvas, 'digit_zone_debug', {}) or {})
        for z in (self.template.zones or []):
            if z.zone_type not in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                continue
            data = self._generate_digit_block_from_model(z)
            if not data:
                continue
            if z.grid:
                z.grid.bubble_positions = list(data['bubble_positions'])
            zone_dbg = dict(dbg.get(z.id, {}))
            zone_dbg.update({
                'anchor_points': data['anchor_points'],
                'anchor_line': data['anchor_line'],
                'col_segments': data['col_lines'],
                'row_segments': data['row_segments'],
                'bubble_centers': data['bubble_centers'],
            })
            dbg[z.id] = zone_dbg
        self.canvas.digit_zone_debug = dbg
        self.canvas.update()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return False
        return self.load_image_from_path(path)

    def load_image_from_path(self, path: str) -> bool:
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid", "Unable to load image")
            return False
        self.template = Template(Path(path).stem, path, pix.width(), pix.height())
        self.template.metadata["alignment_profile"] = "auto"
        self.template.metadata["fill_threshold"] = float(self.fill_threshold_spin.value())
        self.canvas.set_template(self.template, pix)
        self._original_pixmap = pix
        self.template_file_path = None
        self.preview_ok = False
        self.test_ok = False
        self.template_dirty = False
        self._sync_recognition_settings_from_template()
        self._refresh_digit_model_overlay()
        self.result_box.setPlainText("Đã nạp ảnh mẫu. Hãy tạo vùng nhận dạng, preview, test recognition và lưu.")
        return True

    def open_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Template", "", "JSON (*.json)")
        if not path:
            return False
        return self.load_template_from_path(path)

    def load_template_from_path(self, path: str) -> bool:
        try:
            tpl = Template.load_json(path)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid template", f"Unable to load template:\n{exc}")
            return False

        img_path = Path(tpl.image_path)
        if not img_path.is_absolute():
            img_path = (Path(path).parent / img_path).resolve()

        if not img_path.exists():
            new_img, _ = QFileDialog.getOpenFileName(
                self,
                "Template image not found - choose image",
                "",
                "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
            )
            if not new_img:
                return False
            img_path = Path(new_img)

        pix = QPixmap(str(img_path))
        if pix.isNull():
            QMessageBox.warning(self, "Invalid", "Unable to load template image")
            return False

        tpl.image_path = str(img_path)
        tpl.width = pix.width()
        tpl.height = pix.height()
        self.template = tpl
        self._sync_recognition_settings_from_template()
        self.canvas.set_template(self.template, pix)
        self._original_pixmap = pix
        self.template_file_path = path
        self.preview_ok = False
        self.test_ok = False
        self.template_dirty = False
        self._refresh_digit_model_overlay()
        self.result_box.setPlainText("Template loaded. You can preview, adjust and save again.")
        return True

    def regenerate_selected_grid(self, auto: bool = False):
        z = self._selected_zone()
        if not z or z.zone_type not in BLOCK_TYPES:
            if not auto: QMessageBox.warning(self, "No block", "Select semantic block first.")
            return
        z.metadata.update({
            "question_start": self.p_qstart.value(),
            "total_questions": self.p_total.value(),
            "questions_per_block": self.p_qpb.value(),
            "choices_per_question": self.p_choices.value(),
            "questions_per_column": self.p_qpc.value(),
            "column_count": self.p_cols.value(),
            "grid_scale": self.p_scale.value(),
            "bubble_offset_x": self.p_ox.value(),
            "bubble_offset_y": self.p_oy.value(),
            "statements_per_question": self.p_spq.value(),
            "choices_per_statement": self.p_cps.value(),
            "digits_per_answer": self.p_digits.value(),
            "questions": self.p_total.value(),
            "rows": self.p_rows.value(),
            "columns": self.p_columns.value(),
            "digit_map": (
                [int(x.strip()) for x in self.p_digit_map.text().split(",") if x.strip().lstrip("-").isdigit()]
                or (list(range(10)) if z.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else list(range(self.p_rows.value())))
            ),
            "sign_row": self.p_sign_row.value(),
            "decimal_row": self.p_decimal_row.value(),
            "digit_start_row": self.p_digit_start_row.value(),
            "sign_columns": ([int(x.strip()) for x in self.p_sign_columns.text().split(",") if x.strip().lstrip("-").isdigit()] or [1]),
            "decimal_columns": ([int(x.strip()) for x in self.p_decimal_columns.text().split(",") if x.strip().lstrip("-").isdigit()] or [2, 3]),
            "sign_symbol": self.p_sign_symbol.text() or "-",
            "decimal_symbol": self.p_decimal_symbol.text() or ".",
        })
        z.grid = self.template_engine.generate_semantic_grid(z)
        self._refresh_digit_model_overlay()
        self._mark_dirty()
        self.canvas.update()

    def _toggle_add_anchor_mode(self, checked: bool) -> None:
        self.canvas.add_anchor_mode = checked
        if checked and hasattr(self, "digit_anchor_btn"):
            self.digit_anchor_btn.blockSignals(True)
            self.digit_anchor_btn.setChecked(False)
            self.digit_anchor_btn.blockSignals(False)
            self.canvas.add_digit_anchor_mode = False

    def _toggle_add_digit_anchor_mode(self, checked: bool) -> None:
        self.canvas.add_digit_anchor_mode = checked
        if checked:
            self.anchor_btn.blockSignals(True)
            self.anchor_btn.setChecked(False)
            self.anchor_btn.blockSignals(False)
            self.canvas.add_anchor_mode = False
            self.result_box.setPlainText(
                "Chế độ Add Digit Anchor đang bật.\n"
                "- Hãy click vào các marker hình chữ nhật bên phải vùng Student ID / Exam Code.\n"
                "- Mỗi điểm sẽ được lưu dưới tên DIGIT_ANCHOR_XX.\n"
                "- Khi test recognition, các điểm detect được sẽ được đánh dấu X giống anchor góc giấy."
            )

    def generate_digit_zone_anchors(self) -> None:
        """Backward-compatible hook for older QAction wiring.

        Previous revisions exposed a "Generate Digit Anchors" action that called this
        method directly. The current workflow is manual ("Add Digit Anchor"), so this
        compatibility shim simply enables that mode instead of crashing when older UI
        code still calls/ connects `generate_digit_zone_anchors`.
        """
        if hasattr(self, "digit_anchor_btn"):
            self.digit_anchor_btn.setChecked(True)
            return
        self._toggle_add_digit_anchor_mode(True)

    def copy_block(self):
        z = self._selected_zone()
        if z and z.zone_type in BLOCK_TYPES:
            self.clipboard_zone = copy.deepcopy(z)

    def paste_block(self):
        if not self.template or not self.clipboard_zone:
            return
        z = copy.deepcopy(self.clipboard_zone)
        z.id = str(uuid.uuid4())
        z.name = f"{z.name}_paste"
        z.x = min(0.95, z.x + 0.02); z.y = min(0.95, z.y + 0.02)
        self.template.zones.append(z)
        self._mark_dirty()
        self.canvas.update()

    def duplicate_block(self):
        z = self._selected_zone()
        if not z or not self.template:
            return
        self.template.zones.append(self.template_engine.duplicate_zone(z))
        self._mark_dirty()
        self.canvas.update()


    def delete_selected_block(self):
        z = self._selected_zone()
        if not self.template or not z:
            return
        idx = self.canvas.selected_zone
        if 0 <= idx < len(self.template.zones):
            del self.template.zones[idx]
            self._mark_dirty()
            self.canvas.selected_zone = -1
            self.canvas.selection_changed.emit(-1)
            self.canvas.update()

    def delete_selected_anchor(self):
        if not self.template:
            return
        idx = int(getattr(self.canvas, "selected_anchor", -1))
        if idx < 0 or idx >= len(self.template.anchors):
            QMessageBox.information(self, "Delete Anchor", "Chọn anchor cần xoá trước (click vào anchor đen).")
            return
        del self.template.anchors[idx]
        self._mark_dirty()
        self.canvas.selected_anchor = -1
        self.canvas.zones_changed.emit()
        self.canvas.update()

    def snap_grid_to_detected_bubbles(self):
        z = self._selected_zone()
        if not self.template or not z or not z.grid:
            return
        img = cv2.imread(self.template.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        new_pos = []
        for bx, by in z.grid.bubble_positions:
            x, y = int(bx * self.template.width), int(by * self.template.height)
            x0, y0 = max(0, x - 10), max(0, y - 10)
            x1, y1 = min(th.shape[1], x + 11), min(th.shape[0], y + 11)
            roi = th[y0:y1, x0:x1]
            if roi.size == 0 or np.count_nonzero(roi) == 0:
                new_pos.append((bx, by)); continue
            ys, xs = np.where(roi > 0)
            new_pos.append(((x0 + float(xs.mean())) / self.template.width, (y0 + float(ys.mean())) / self.template.height))
        z.grid.bubble_positions = new_pos
        self._mark_dirty()
        self.canvas.update()

    def preview_template(self):
        if not self.template:
            return
        errors = self._validate_template()
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors)); self.preview_ok = False; return
        self.canvas.preview_mode = True
        self.canvas.update()
        self.preview_ok = True

    def _validate_template(self) -> list[str]:
        if not self.template: return ["No template loaded."]
        errs = self.template_engine.validate_template(self.template)
        for z in self.template.zones:
            if z.zone_type in BLOCK_TYPES and not z.grid:
                errs.append(f"{z.name}: missing semantic grid.")
        return errs

    def test_recognition(self):
        if not self.template or not self.preview_ok:
            QMessageBox.warning(self, "Workflow", "Load, configure blocks, and preview first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load scanned sheet", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return

        self._apply_recognition_settings_to_engine()
        res = self.omr.run_recognition_test(path, self.template)
        aligned = getattr(res, "aligned_image", None)
        aligned_binary = getattr(res, "aligned_binary", None)
        if aligned is None or aligned_binary is None:
            QMessageBox.warning(self, "Recognition", "Không thể căn chỉnh ảnh theo template để test recognition.")
            return

        # Show what engine actually uses for recognition.
        bg = self._cv_to_qpixmap(aligned)
        if not bg.isNull() and self.template:
            self.canvas.pixmap = bg
            self.canvas.resize(int(bg.width() * self.canvas.zoom), int(bg.height() * self.canvas.zoom))

        self.result_box.setPlainText(
            f"Student ID: {res.student_id or '-'}\nExam Code: {res.exam_code or '-'}\n"
            f"MCQ: {', '.join([f'Q{k}:{v}' for k, v in sorted(res.mcq_answers.items())]) or '(none)'}\n"
            f"TF: {res.true_false_answers or {}}\n"
            f"NUM: {res.numeric_answers or {}}"
        )

        # Show detected anchors and recognized options overlay (green X = selected/seen).
        self.canvas.recognition_overlay.clear()
        self.canvas.detected_anchor_points = list(getattr(res, "detected_anchors", []))
        self.canvas.digit_zone_debug = dict(getattr(res, "digit_zone_debug", {}) or {})
        self.canvas.recognition_overlay.update(getattr(res, "bubble_states_by_zone", {}) or self.omr.extract_bubble_states(aligned_binary, self.template))

        self.result_box.append(f"\nDetected anchors: {len(self.canvas.detected_anchor_points)}")

        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_ok = True

    @staticmethod
    def _cv_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
        if image_bgr is None or image_bgr.size == 0:
            return QPixmap()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def save_template(self):
        return self._save_template(save_as=False)

    def save_template_as(self):
        return self._save_template(save_as=True)

    def _save_template(self, save_as: bool):
        if not self.template:
            return False
        if not self.preview_ok:
            QMessageBox.warning(self, "Preview required", "Run preview first.")
            return False
        if not self.test_ok:
            QMessageBox.warning(self, "Test required", "Run test recognition first.")
            return False
        errs = self._validate_template()
        if errs:
            QMessageBox.warning(self, "Validation", "\n".join(errs))
            return False
        path = self.template_file_path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save Template", path or "template.json", "JSON (*.json)")
        if path:
            if self.template is not None:
                self.template.metadata["alignment_profile"] = str(self.align_profile_combo.currentData() or "auto")
                self.template.metadata["fill_threshold"] = float(self.fill_threshold_spin.value())
                self.template.name = str(Path(path).stem)
            self.template.save_json(path)
            self.template_file_path = path
            self.template_dirty = False
            if callable(self.on_template_saved):
                self.on_template_saved(path, self.template.name if self.template else Path(path).stem)
            return True
        return False

    def _mark_dirty(self) -> None:
        if self.template is not None:
            self.template_dirty = True

    def has_unsaved_changes(self) -> bool:
        return bool(self.template is not None and self.template_dirty)

    def _confirm_close(self) -> bool:
        if not self.has_unsaved_changes():
            return True
        msg = QMessageBox(self)
        msg.setWindowTitle("Template Editor")
        msg.setText("Mẫu giấy thi có thay đổi chưa lưu.")
        msg.setInformativeText("Bạn có muốn lưu trước khi đóng không?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Save)
        choice = msg.exec()
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            return bool(self.save_template())
        return True

    def closeEvent(self, event):  # type: ignore[override]
        if self._confirm_close():
            event.accept()
            return
        event.ignore()

    def _on_alignment_profile_changed(self, _idx: int) -> None:
        if self.template is None:
            return
        self.template.metadata["alignment_profile"] = str(self.align_profile_combo.currentData() or "auto")
        self._mark_dirty()

    def _on_fill_threshold_changed(self, value: float) -> None:
        if self.template is None:
            return
        self.template.metadata["fill_threshold"] = float(max(0.05, min(0.95, float(value))))
        self._mark_dirty()

    def _apply_recognition_settings_to_engine(self) -> None:
        self.omr.alignment_profile = str(self.align_profile_combo.currentData() or "auto")
        self.omr.fill_threshold = float(max(0.05, min(0.95, float(self.fill_threshold_spin.value()))))

    def _sync_recognition_settings_from_template(self) -> None:
        self._sync_alignment_profile_from_template()
        if not hasattr(self, "fill_threshold_spin"):
            return
        value = 0.45
        if self.template is not None:
            try:
                value = float((self.template.metadata or {}).get("fill_threshold", 0.45) or 0.45)
            except Exception:
                value = 0.45
        value = max(0.05, min(0.95, value))
        self.fill_threshold_spin.blockSignals(True)
        self.fill_threshold_spin.setValue(value)
        self.fill_threshold_spin.blockSignals(False)
        if self.template is not None:
            self.template.metadata["fill_threshold"] = value

    def _sync_alignment_profile_from_template(self) -> None:
        if self.template is None or not hasattr(self, "align_profile_combo"):
            return
        mode = str((self.template.metadata or {}).get("alignment_profile", "auto") or "auto")
        idx = self.align_profile_combo.findData(mode)
        if idx < 0:
            idx = 0
        self.align_profile_combo.setCurrentIndex(idx)

    def run_template_quality_check(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "Template QC", "Load a template/image first.")
            return
        issues: list[str] = []
        tips: list[str] = []
        anchors = list(self.template.anchors or [])
        if len(anchors) < 4:
            issues.append("Anchor count < 4 (không đủ để homography ổn định).")
        else:
            mode = str((self.template.metadata or {}).get("alignment_profile", "auto") or "auto")
            max_anchor = 120 if mode == "one_side" else 60
            if len(anchors) > max_anchor:
                issues.append(f"Anchor count quá nhiều (>{max_anchor}), dễ bắt nhầm marker nhiễu.")

        w, h = float(self.template.width), float(self.template.height)
        if anchors:
            near_edge = 0
            for a in anchors:
                ax = a.x * w if a.x <= 1.0 else a.x
                ay = a.y * h if a.y <= 1.0 else a.y
                if ax < w * 0.08 or ay < h * 0.08 or ax > w * 0.92 or ay > h * 0.92:
                    near_edge += 1
            if near_edge >= max(2, len(anchors) // 2):
                tips.append("Anchor nằm sát biên nhiều: chọn Alignment=Border.")
                xs = [float((a.x * w if a.x <= 1.0 else a.x)) for a in anchors]
                if (max(xs) - min(xs)) <= w * 0.18:
                    tips.append("Anchor gần như 1 cột dọc: chọn Alignment=One-side ruler để chuẩn hóa theo các dòng mốc.")
            else:
                tips.append("Anchor phân bố trung tâm/ổn định: nên chọn Alignment=Legacy hoặc Auto.")

        block_zones = [z for z in self.template.zones if z.zone_type in BLOCK_TYPES]
        if not block_zones:
            issues.append("Chưa có block nhận dạng (MCQ/TF/NUM/ID/EXAM_CODE).")
        missing_grid = [z.name for z in block_zones if not z.grid]
        if missing_grid:
            issues.append(f"Thiếu semantic grid: {', '.join(missing_grid[:6])}{'...' if len(missing_grid) > 6 else ''}")

        overlap_count = 0
        for i in range(len(block_zones)):
            a = block_zones[i]
            ar = (a.x, a.y, a.x + a.width, a.y + a.height)
            for j in range(i + 1, len(block_zones)):
                b = block_zones[j]
                br = (b.x, b.y, b.x + b.width, b.y + b.height)
                ix = max(0.0, min(ar[2], br[2]) - max(ar[0], br[0]))
                iy = max(0.0, min(ar[3], br[3]) - max(ar[1], br[1]))
                if ix * iy > 1e-4:
                    overlap_count += 1
        if overlap_count > 0:
            issues.append(f"Có {overlap_count} vùng block bị overlap.")

        score = 100
        score -= min(40, len(issues) * 12)
        score = max(0, score)
        msg = [f"Template QC score: {score}/100", "", "Issues:"]
        msg.extend([f"- {x}" for x in issues] if issues else ["- Không phát hiện lỗi nghiêm trọng."])
        msg.append("")
        msg.append("Recommendations:")
        msg.extend([f"- {x}" for x in tips] if tips else ["- Dùng Alignment=Auto."])
        self.result_box.setPlainText("\n".join(msg))
