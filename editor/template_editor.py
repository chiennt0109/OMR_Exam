from __future__ import annotations

import copy
from pathlib import Path
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
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
    QSpinBox,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.omr_engine import OMRProcessor
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
        self.current_zone_type = ZoneType.MCQ_BLOCK
        self.selected_zone = -1

        self._drawing = False
        self._moving = False
        self._drag_corner = -1
        self._start = QPoint()
        self._current_rect = QRect()
        self._move_offset = QPoint()

        self.recognition_overlay: dict[str, list[bool]] = {}
        self.setFocusPolicy(Qt.StrongFocus)

    def set_template(self, template: Template, pixmap: QPixmap):
        self.template = template
        self.pixmap = pixmap
        self.zoom = 1.0
        self.selected_zone = -1
        self.preview_mode = False
        self.recognition_overlay.clear()
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


    def keyPressEvent(self, e: QKeyEvent):
        if not self.template:
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and 0 <= self.selected_zone < len(self.template.zones):
            del self.template.zones[self.selected_zone]
            self.selected_zone = -1
            self.selection_changed.emit(-1)
            self.zones_changed.emit()
            self.update()

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
        for a in self.template.anchors:
            p.drawRect(QRectF(a.x * self.template.width * self.zoom - 4, a.y * self.template.height * self.zoom - 4, 8, 8))

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
            color = QColor(0, 180, 0) if (i < len(states) and states[i]) else QColor(230, 60, 60)
            painter.setPen(QPen(color, 1)); painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(bx * self.template.width * self.zoom, by * self.template.height * self.zoom), 4, 4)


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1620, 940)

        self.template_engine = TemplateEngine()
        self.omr = OMRProcessor()
        self.template: Template | None = None
        self.clipboard_zone: Zone | None = None
        self.preview_ok = False
        self.test_ok = False
        self._sync = False

        self.canvas = TemplateCanvas()
        self.canvas.selection_changed.connect(self._load_props)
        self.canvas.zones_changed.connect(self._on_zone_changed)

        scroll = QScrollArea(); scroll.setWidgetResizable(False); scroll.setWidget(self.canvas)

        toolbar = QToolBar("Template")
        self.addToolBar(toolbar)
        for label, fn in [
            ("Load Blank Paper", self.load_image),
            ("Preview", self.preview_template),
            ("Test Recognition", self.test_recognition),
            ("Save Template", self.save_template),
            ("Copy Block", self.copy_block),
            ("Paste Block", self.paste_block),
            ("Duplicate Block", self.duplicate_block),
            ("Delete Block", self.delete_selected_block),
            ("Snap Grid", self.snap_grid_to_detected_bubbles),
        ]:
            a = QAction(label, self); a.triggered.connect(fn); toolbar.addAction(a)

        self.anchor_btn = QPushButton("Add Anchor"); self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(lambda c: setattr(self.canvas, "add_anchor_mode", c))
        toolbar.addWidget(self.anchor_btn)

        toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_type = QComboBox(); self.zone_type.addItems([z.value for z in [ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK, ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK]])
        self.zone_type.currentTextChanged.connect(lambda t: setattr(self.canvas, "current_zone_type", ZoneType(t)))
        toolbar.addWidget(self.zone_type)

        zin = QAction("Zoom +", self); zin.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15)); toolbar.addAction(zin)
        zout = QAction("Zoom -", self); zout.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15)); toolbar.addAction(zout)

        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMaximumHeight(160)
        self.prop_panel = self._build_prop_panel()

        center = QWidget(); layout = QHBoxLayout(center)
        left = QVBoxLayout(); left.addWidget(scroll); left.addWidget(self.result_box)
        layout.addLayout(left, 1); layout.addWidget(self.prop_panel)
        self.setCentralWidget(center)

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

        self._prop_controls = [
            ("question_start", self.p_qstart), ("total_questions", self.p_total), ("choices_per_question", self.p_choices),
            ("questions_per_column", self.p_qpc), ("column_count", self.p_cols), ("grid_scale", self.p_scale),
            ("offset_x", self.p_ox), ("offset_y", self.p_oy), ("questions_per_block", self.p_qpb),
            ("statements_per_question", self.p_spq), ("choices_per_statement", self.p_cps), ("digits_per_answer", self.p_digits), ("rows", self.p_rows), ("columns", self.p_columns), ("digit_map", self.p_digit_map),
        ]
        self._prop_rows = {}
        for name, widget in self._prop_controls:
            lbl = QLabel(name)
            f.addRow(lbl, widget)
            self._prop_rows[name] = (lbl, widget)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._prop_changed)
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
        if zone_type == ZoneType.MCQ_BLOCK:
            visible.update({"questions_per_block": True, "choices_per_question": True})
        elif zone_type == ZoneType.TRUE_FALSE_BLOCK:
            visible.update({"questions_per_block": True, "statements_per_question": True, "choices_per_statement": True})
        elif zone_type == ZoneType.NUMERIC_BLOCK:
            visible.update({"questions_per_block": True, "digits_per_answer": True, "rows": True, "digit_map": True})
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
        if self.canvas.preview_mode:
            self.regenerate_selected_grid(auto=True)
            self.preview_template()

    def _load_props(self, _idx: int):
        z = self._selected_zone()
        enabled = bool(z and z.zone_type in BLOCK_TYPES)
        for w in [self.p_qstart, self.p_total, self.p_choices, self.p_qpc, self.p_cols, self.p_scale, self.p_ox, self.p_oy, self.p_qpb, self.p_spq, self.p_cps, self.p_digits, self.p_rows, self.p_columns, self.p_digit_map, self.btn_regen]:
            w.setEnabled(enabled)
        self._apply_block_property_visibility(z.zone_type if z else None)
        if not enabled:
            return
        md = z.metadata
        self._sync = True
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
        self.p_rows.setValue(int(md.get("rows", 10)))
        self.p_columns.setValue(int(md.get("columns", 8)))
        self.p_digit_map.setText(",".join(str(x) for x in md.get("digit_map", list(range(10)))))
        self._sync = False

    def _prop_changed(self, *_args):
        if not self._sync:
            self.regenerate_selected_grid(auto=True)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid", "Unable to load image")
            return
        self.template = Template(Path(path).stem, path, pix.width(), pix.height())
        self.canvas.set_template(self.template, pix)
        self.preview_ok = False; self.test_ok = False

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
            "digit_map": ([int(x.strip()) for x in self.p_digit_map.text().split(",") if x.strip().lstrip("-").isdigit()] or list(range(self.p_rows.value()))),
        })
        z.grid = self.template_engine.generate_semantic_grid(z)
        self.canvas.update()

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
        self.canvas.update()

    def duplicate_block(self):
        z = self._selected_zone()
        if not z or not self.template:
            return
        self.template.zones.append(self.template_engine.duplicate_zone(z))
        self.canvas.update()


    def delete_selected_block(self):
        z = self._selected_zone()
        if not self.template or not z:
            return
        idx = self.canvas.selected_zone
        if 0 <= idx < len(self.template.zones):
            del self.template.zones[idx]
            self.canvas.selected_zone = -1
            self.canvas.selection_changed.emit(-1)
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

        res = self.omr.process_image(path, self.template)
        self.result_box.setPlainText(
            f"Student ID: {res.student_id or '-'}\nExam Code: {res.exam_code or '-'}\n"
            f"MCQ: {', '.join([f'Q{k}:{v}' for k, v in sorted(res.mcq_answers.items())]) or '(none)'}\n"
            f"TF: {res.true_false_answers or {}}\n"
            f"NUM: {res.numeric_answers or {}}"
        )

        # generate overlay by re-measuring bubble fills
        self.canvas.recognition_overlay.clear()
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
            for z in self.template.zones:
                if not z.grid:
                    continue
                states = []
                for bx, by in z.grid.bubble_positions:
                    x, y = int(bx * self.template.width), int(by * self.template.height)
                    x0, y0 = max(0, x - 6), max(0, y - 6)
                    x1, y1 = min(th.shape[1], x + 6), min(th.shape[0], y + 6)
                    roi = th[y0:y1, x0:x1]
                    states.append(False if roi.size == 0 else (float(np.count_nonzero(roi)) / float(roi.size) >= self.omr.fill_threshold))
                self.canvas.recognition_overlay[z.id] = states

        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_ok = True

    def save_template(self):
        if not self.template:
            return
        if not self.preview_ok:
            QMessageBox.warning(self, "Preview required", "Run preview first.")
            return
        if not self.test_ok:
            QMessageBox.warning(self, "Test required", "Run test recognition first.")
            return
        errs = self._validate_template()
        if errs:
            QMessageBox.warning(self, "Validation", "\n".join(errs)); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if path:
            self.template.save_json(path)
