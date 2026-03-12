from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from core.omr_engine import OMRProcessor
from core.template_engine import TemplateEngine
from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


@dataclass
class GridParams:
    total_questions: int
    choices_per_question: int
    questions_per_column: int
    column_count: int
    scale: float
    offset_x: float
    offset_y: float


BLOCK_TYPES = {ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK, ZoneType.ID_BLOCK}


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
        self.active_corner_idx = -1
        self.recognition_overlay: dict[str, list[bool]] = {}

        self._drawing = False
        self._moving_zone = False
        self._resizing_zone = False
        self._drag_corner = -1
        self._start = QPoint()
        self._current_rect = QRect()
        self._move_offset = QPoint()

        self.setMouseTracking(True)

    def set_template(self, template: Template, pixmap: QPixmap) -> None:
        self.template = template
        self.pixmap = pixmap
        self.zoom = 1.0
        self.selected_zone = -1
        self.active_corner_idx = -1
        self._update_canvas_size()
        self.update()

    def _update_canvas_size(self):
        if self.pixmap:
            self.resize(int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))

    def set_zoom(self, z: float):
        self.zoom = max(0.25, min(4.0, z))
        self._update_canvas_size()
        self.update()

    def _to_img(self, p: QPoint) -> QPoint:
        return QPoint(int(p.x() / self.zoom), int(p.y() / self.zoom))

    def _zone_abs_rect(self, z: Zone) -> QRect:
        assert self.template
        return QRect(int(z.x * self.template.width), int(z.y * self.template.height), max(1, int(z.width * self.template.width)), max(1, int(z.height * self.template.height)))

    def _zone_corners_abs(self, z: Zone) -> list[QPointF]:
        assert self.template
        corners = z.metadata.get("corners")
        if not corners:
            return [
                QPointF(z.x * self.template.width, z.y * self.template.height),
                QPointF((z.x + z.width) * self.template.width, z.y * self.template.height),
                QPointF(z.x * self.template.width, (z.y + z.height) * self.template.height),
                QPointF((z.x + z.width) * self.template.width, (z.y + z.height) * self.template.height),
            ]
        return [QPointF(c[0] * self.template.width, c[1] * self.template.height) for c in corners]

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if not self.template or not self.pixmap or e.button() != Qt.LeftButton:
            return
        p = self._to_img(e.position().toPoint())

        if self.add_anchor_mode:
            self.template.anchors.append(AnchorPoint(p.x() / self.template.width, p.y() / self.template.height, f"A{len(self.template.anchors)+1}"))
            self.zones_changed.emit(); self.update(); return

        hit, corner = self._hit_zone_or_corner(p)
        if hit >= 0:
            self.selected_zone = hit
            self.selection_changed.emit(hit)
            if corner >= 0:
                self._drag_corner = corner
                self.active_corner_idx = corner
            else:
                self._moving_zone = True
                zr = self._zone_abs_rect(self.template.zones[hit])
                self._move_offset = QPoint(p.x() - zr.x(), p.y() - zr.y())
            self.update(); return

        self.selected_zone = -1
        self.selection_changed.emit(-1)
        self._drawing = True
        self._start = p
        self._current_rect = QRect(p, p)
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self.template:
            return
        p = self._to_img(e.position().toPoint())
        if self._drawing:
            self._current_rect = QRect(self._start, p).normalized()
            self.update(); return

        if self.selected_zone < 0:
            return
        z = self.template.zones[self.selected_zone]

        if self._moving_zone:
            zr = self._zone_abs_rect(z)
            zr.moveTo(p.x() - self._move_offset.x(), p.y() - self._move_offset.y())
            z.x = zr.x() / self.template.width
            z.y = zr.y() / self.template.height
            self._translate_corners_with_zone(z)
            self.zones_changed.emit(); self.update(); return

        if self._drag_corner >= 0:
            corners = self._zone_corners_abs(z)
            corners[self._drag_corner] = QPointF(p.x(), p.y())
            z.metadata["corners"] = [[c.x() / self.template.width, c.y() / self.template.height] for c in corners]
            self.zones_changed.emit(); self.update()

    def _translate_corners_with_zone(self, z: Zone) -> None:
        # when zone moves, keep relative corner shape by translating all corners by delta from bounding box
        if "corners" not in z.metadata:
            return
        corners = self._zone_corners_abs(z)
        xs, ys = [c.x() for c in corners], [c.y() for c in corners]
        bx, by = min(xs), min(ys)
        target_x = z.x * self.template.width
        target_y = z.y * self.template.height
        dx, dy = target_x - bx, target_y - by
        moved = [QPointF(c.x() + dx, c.y() + dy) for c in corners]
        z.metadata["corners"] = [[c.x() / self.template.width, c.y() / self.template.height] for c in moved]

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if not self.template or e.button() != Qt.LeftButton:
            return
        if self._drawing:
            self._drawing = False
            r = self._current_rect.normalized()
            if r.width() > 8 and r.height() > 8:
                z = Zone(str(uuid.uuid4()), f"{self.current_zone_type.value}_{len(self.template.zones)+1}", self.current_zone_type,
                         r.x() / self.template.width, r.y() / self.template.height, r.width() / self.template.width, r.height() / self.template.height,
                         metadata={})
                z.metadata["corners"] = [
                    [z.x, z.y],
                    [z.x + z.width, z.y],
                    [z.x, z.y + z.height],
                    [z.x + z.width, z.y + z.height],
                ]
                self.template.zones.append(z)
                self.selected_zone = len(self.template.zones) - 1
                self.selection_changed.emit(self.selected_zone)
                self.zones_changed.emit()
            self._current_rect = QRect()

        self._moving_zone = False
        self._drag_corner = -1
        self.active_corner_idx = -1
        self.update()

    def _hit_zone_or_corner(self, p: QPoint) -> tuple[int, int]:
        if not self.template:
            return -1, -1
        for i in range(len(self.template.zones) - 1, -1, -1):
            z = self.template.zones[i]
            corners = self._zone_corners_abs(z)
            for ci, c in enumerate(corners):
                if abs(c.x() - p.x()) <= 8 and abs(c.y() - p.y()) <= 8:
                    return i, ci
            if self._zone_abs_rect(z).contains(p):
                return i, -1
        return -1, -1

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
            p.drawEllipse(QPointF(a.x * self.template.width * self.zoom, a.y * self.template.height * self.zoom), 4, 4)

        for i, z in enumerate(self.template.zones):
            zr = self._zone_abs_rect(z)
            p.setPen(QPen(QColor(220, 60, 60), 2)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(zr.x() * self.zoom, zr.y() * self.zoom, zr.width() * self.zoom, zr.height() * self.zoom))
            corners = self._zone_corners_abs(z)
            p.setPen(QPen(QColor(30, 120, 250), 1)); p.setBrush(QColor(30, 120, 250))
            for c in corners:
                p.drawEllipse(QPointF(c.x() * self.zoom, c.y() * self.zoom), 4, 4)

            if self.preview_mode and z.grid:
                self._draw_preview_grid(p, z)
            if z.id in self.recognition_overlay and z.grid:
                self._draw_recognition(p, z)

        if self._drawing:
            r = self._current_rect
            p.setPen(QPen(Qt.green, 2, Qt.DashLine)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(r.x() * self.zoom, r.y() * self.zoom, r.width() * self.zoom, r.height() * self.zoom))

    def _draw_preview_grid(self, painter: QPainter, z: Zone):
        assert self.template and z.grid
        choices = max(1, len(z.grid.options))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i, (bx, by) in enumerate(z.grid.bubble_positions):
            x, y = bx * self.template.width * self.zoom, by * self.template.height * self.zoom
            painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
            q_idx = i // choices
            c_idx = i % choices
            if c_idx == 0:
                painter.drawText(QPointF(x - 22, y + 4), str(z.grid.question_start + q_idx))
            painter.drawText(QPointF(x + 4, y - 2), z.grid.options[c_idx])

    def _draw_recognition(self, painter: QPainter, z: Zone):
        assert self.template and z.grid
        states = self.recognition_overlay[z.id]
        for i, (bx, by) in enumerate(z.grid.bubble_positions):
            painter.setPen(QPen(QColor(0, 170, 0) if (i < len(states) and states[i]) else QColor(225, 60, 60), 1))
            painter.drawEllipse(QPointF(bx * self.template.width * self.zoom, by * self.template.height * self.zoom), 4, 4)


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1620, 940)

        self.template_engine = TemplateEngine()
        self.omr = OMRProcessor()
        self.template: Template | None = None
        self.preview_ok = False
        self.test_ok = False
        self.clipboard_zone: Zone | None = None
        self._sync = False

        self.canvas = TemplateCanvas()
        self.canvas.selection_changed.connect(self._load_selected_zone_props)
        self.canvas.zones_changed.connect(self._on_zone_changed)

        scroll = QScrollArea(); scroll.setWidgetResizable(False); scroll.setWidget(self.canvas)

        toolbar = QToolBar("Tools"); self.addToolBar(toolbar)
        for text, slot in [
            ("Load Blank Paper", self.load_image),
            ("Preview", self.preview_template),
            ("Test Recognition", self.test_recognition),
            ("Save Template JSON", self.save_template),
            ("Copy Block", self.copy_block),
            ("Paste Block", self.paste_block),
            ("Duplicate Block", self.duplicate_block),
            ("Snap Grid", self.snap_grid_to_detected_bubbles),
        ]:
            a = QAction(text, self); a.triggered.connect(slot); toolbar.addAction(a)

        self.anchor_btn = QPushButton("Add Anchor"); self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(lambda c: setattr(self.canvas, "add_anchor_mode", c))
        toolbar.addWidget(self.anchor_btn)

        self.zone_type = QComboBox(); self.zone_type.addItems([z.value for z in [ZoneType.STUDENT_ID, ZoneType.EXAM_CODE, ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK, ZoneType.ID_BLOCK]])
        self.zone_type.currentTextChanged.connect(lambda t: setattr(self.canvas, "current_zone_type", ZoneType(t)))
        toolbar.addWidget(QLabel(" Zone Type: ")); toolbar.addWidget(self.zone_type)

        zin = QAction("Zoom +", self); zin.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15)); toolbar.addAction(zin)
        zout = QAction("Zoom -", self); zout.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15)); toolbar.addAction(zout)

        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMaximumHeight(150)
        self.props = self._build_prop_panel()

        center = QWidget(); lay = QHBoxLayout(center)
        left = QVBoxLayout(); left.addWidget(scroll); left.addWidget(self.result_box)
        lay.addLayout(left, 1); lay.addWidget(self.props)
        self.setCentralWidget(center)

    def _build_prop_panel(self) -> QWidget:
        w = QWidget(); w.setFixedWidth(340); l = QVBoxLayout(w)
        l.addWidget(QLabel("Grid Properties (shared engine)"))
        f = QFormLayout()
        self.p_total = QSpinBox(); self.p_total.setRange(1, 2000); self.p_total.setValue(40)
        self.p_choices = QSpinBox(); self.p_choices.setRange(2, 20); self.p_choices.setValue(4)
        self.p_qpc = QSpinBox(); self.p_qpc.setRange(1, 500); self.p_qpc.setValue(10)
        self.p_cols = QSpinBox(); self.p_cols.setRange(1, 20); self.p_cols.setValue(4)
        self.p_scale = QDoubleSpinBox(); self.p_scale.setRange(0.1, 3.0); self.p_scale.setSingleStep(0.05); self.p_scale.setValue(1.0)
        self.p_offx = QDoubleSpinBox(); self.p_offx.setRange(-1.0, 1.0); self.p_offx.setSingleStep(0.01)
        self.p_offy = QDoubleSpinBox(); self.p_offy.setRange(-1.0, 1.0); self.p_offy.setSingleStep(0.01)

        f.addRow("total_questions", self.p_total)
        f.addRow("choices_per_question", self.p_choices)
        f.addRow("questions_per_column", self.p_qpc)
        f.addRow("column_count", self.p_cols)
        f.addRow("grid_scale", self.p_scale)
        f.addRow("bubble_offset_x", self.p_offx)
        f.addRow("bubble_offset_y", self.p_offy)
        l.addLayout(f)
        self.btn_regen = QPushButton("Regenerate Grid"); self.btn_regen.clicked.connect(self.regenerate_selected_grid)
        l.addWidget(self.btn_regen); l.addStretch(1)

        for s in [self.p_total, self.p_choices, self.p_qpc, self.p_cols, self.p_scale, self.p_offx, self.p_offy]:
            s.valueChanged.connect(self._prop_changed)
        return w

    def _selected(self) -> Zone | None:
        if not self.template: return None
        i = self.canvas.selected_zone
        if i < 0 or i >= len(self.template.zones): return None
        return self.template.zones[i]

    def _on_zone_changed(self):
        if self.canvas.preview_mode:
            self.regenerate_selected_grid(auto=True)
            self.preview_template()

    def _load_selected_zone_props(self, idx: int):
        z = self._selected()
        enabled = bool(z and z.zone_type in BLOCK_TYPES)
        for w in [self.p_total, self.p_choices, self.p_qpc, self.p_cols, self.p_scale, self.p_offx, self.p_offy, self.btn_regen]:
            w.setEnabled(enabled)
        if not enabled:
            return
        self._sync = True
        md = z.metadata
        self.p_total.setValue(int(md.get("total_questions", 40)))
        self.p_choices.setValue(int(md.get("choices_per_question", 4)))
        self.p_qpc.setValue(int(md.get("questions_per_column", 10)))
        self.p_cols.setValue(int(md.get("column_count", 4)))
        self.p_scale.setValue(float(md.get("grid_scale", 1.0)))
        self.p_offx.setValue(float(md.get("bubble_offset_x", 0.0)))
        self.p_offy.setValue(float(md.get("bubble_offset_y", 0.0)))
        self._sync = False

    def _prop_changed(self):
        if not self._sync:
            self.regenerate_selected_grid(auto=True)

    def load_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not p: return
        pix = QPixmap(p)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid", "Unable to load image"); return
        self.template = Template(Path(p).stem, p, pix.width(), pix.height())
        self.canvas.set_template(self.template, pix)
        self.preview_ok = self.test_ok = False

    def regenerate_selected_grid(self, auto: bool = False):
        z = self._selected()
        if not z or z.zone_type not in BLOCK_TYPES:
            if not auto: QMessageBox.warning(self, "No block", "Select MCQ/TRUE_FALSE/NUMERIC/ID block zone.")
            return
        params = GridParams(self.p_total.value(), self.p_choices.value(), self.p_qpc.value(), self.p_cols.value(), self.p_scale.value(), self.p_offx.value(), self.p_offy.value())
        z.metadata.update({
            "total_questions": params.total_questions,
            "choices_per_question": params.choices_per_question,
            "questions_per_column": params.questions_per_column,
            "column_count": params.column_count,
            "grid_scale": params.scale,
            "bubble_offset_x": params.offset_x,
            "bubble_offset_y": params.offset_y,
        })
        z.grid = self._generate_deformable_grid(z, params)
        self.canvas.update()
        if self.canvas.preview_mode:
            self.preview_template()

    def _default_choices_for_type(self, zt: ZoneType, requested: int) -> list[str]:
        if zt == ZoneType.TRUE_FALSE_BLOCK:
            return ["T", "F"]
        if zt in (ZoneType.NUMERIC_BLOCK, ZoneType.ID_BLOCK):
            return [str(i) for i in range(10)]
        return [chr(65 + i) for i in range(min(requested, 26))]

    def _generate_deformable_grid(self, z: Zone, p: GridParams) -> BubbleGrid:
        assert self.template
        choices = self._default_choices_for_type(z.zone_type, p.choices_per_question)
        q_count = max(1, p.total_questions)
        qpc = max(1, p.questions_per_column)
        cols = max(1, p.column_count)

        corners_rel = z.metadata.get("corners")
        if not corners_rel:
            corners_rel = [[z.x, z.y], [z.x + z.width, z.y], [z.x, z.y + z.height], [z.x + z.width, z.y + z.height]]
            z.metadata["corners"] = corners_rel
        tl, tr, bl, br = [np.array(c, dtype=float) for c in corners_rel]

        bubbles: list[tuple[float, float]] = []
        for q in range(q_count):
            col = q // qpc
            row = q % qpc
            if col >= cols: break
            u = (col + 0.5) / cols
            v = (row + 0.5) / qpc
            for c in range(len(choices)):
                du = ((c + 0.5) / len(choices) - 0.5) * (1.0 / cols) * p.scale
                uu = min(0.999, max(0.001, u + du + p.offset_x))
                vv = min(0.999, max(0.001, v + p.offset_y))
                pt = (1 - uu) * (1 - vv) * tl + uu * (1 - vv) * tr + (1 - uu) * vv * bl + uu * vv * br
                bubbles.append((float(pt[0]), float(pt[1])))

        actual_q = len(bubbles) // len(choices)
        return BubbleGrid(rows=qpc, cols=cols, question_start=1, question_count=actual_q, options=choices, bubble_positions=bubbles)

    def copy_block(self):
        z = self._selected()
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
        self.canvas.selected_zone = len(self.template.zones) - 1
        self.canvas.selection_changed.emit(self.canvas.selected_zone)
        self.canvas.update()

    def duplicate_block(self):
        z = self._selected()
        if not self.template or not z:
            return
        dup = copy.deepcopy(z)
        dup.id = str(uuid.uuid4())
        dup.name = f"{z.name}_copy"
        dup.x = min(0.95, dup.x + 0.015); dup.y = min(0.95, dup.y + 0.015)
        self.template.zones.append(dup)
        self.canvas.update()

    def snap_grid_to_detected_bubbles(self):
        z = self._selected()
        if not self.template or not z or not z.grid:
            return
        # pragmatic snap: nudge bubbles to nearest dark pixel centroid in local window from template image
        img = cv2.imread(self.template.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        new_positions = []
        for bx, by in z.grid.bubble_positions:
            x, y = int(bx * self.template.width), int(by * self.template.height)
            x0, y0, x1, y1 = max(0, x - 8), max(0, y - 8), min(th.shape[1], x + 9), min(th.shape[0], y + 9)
            roi = th[y0:y1, x0:x1]
            if roi.size == 0 or np.count_nonzero(roi) == 0:
                new_positions.append((bx, by)); continue
            ys, xs = np.where(roi > 0)
            cx = x0 + float(xs.mean()); cy = y0 + float(ys.mean())
            new_positions.append((cx / self.template.width, cy / self.template.height))
        z.grid.bubble_positions = new_positions
        self.canvas.update()

    def preview_template(self):
        if not self.template:
            return
        errs = self._validate()
        if errs:
            QMessageBox.warning(self, "Validation", "\n".join(errs)); self.preview_ok = False; return
        self.canvas.preview_mode = True
        self.canvas.update()
        self.preview_ok = True

    def _normalize_to_200_dpi(self, path: str) -> tuple[str, str]:
        q = QImage(path)
        if q.isNull(): return path, ""
        dpi = int(round(q.dotsPerMeterX() * 0.0254)) if q.dotsPerMeterX() > 0 else 0
        if dpi == 200: return path, ""
        if dpi <= 0: return path, "Image DPI metadata missing; expected 200 DPI."
        im = cv2.imread(path)
        if im is None: return path, ""
        s = 200.0 / dpi
        out = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
        p = str(Path(path).with_name(f"{Path(path).stem}_200dpi.png"))
        cv2.imwrite(p, out)
        return p, f"Input DPI={dpi}. Normalized to 200 DPI scale."

    def _to_abs_template(self) -> Template:
        assert self.template
        t = Template(self.template.name, self.template.image_path, self.template.width, self.template.height)
        t.anchors = [AnchorPoint(a.x * t.width, a.y * t.height, a.name) for a in self.template.anchors]
        for z in self.template.zones:
            g = None
            if z.grid:
                g = BubbleGrid(z.grid.rows, z.grid.cols, z.grid.question_start, z.grid.question_count, z.grid.options, [(bx * t.width, by * t.height) for bx, by in z.grid.bubble_positions])
            t.zones.append(Zone(z.id, z.name, z.zone_type, z.x * t.width, z.y * t.height, z.width * t.width, z.height * t.height, g, copy.deepcopy(z.metadata)))
        return t

    def test_recognition(self):
        if not self.template or not self.preview_ok:
            QMessageBox.warning(self, "Workflow", "Load image and preview first."); return
        p, _ = QFileDialog.getOpenFileName(self, "Load Scanned", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not p: return
        npath, msg = self._normalize_to_200_dpi(p)
        if msg: QMessageBox.warning(self, "DPI Check", msg)

        t = self._to_abs_template()
        res = self.omr.process_image(npath, t)
        gray = cv2.imread(npath, cv2.IMREAD_GRAYSCALE)
        if gray is None: return
        corrected = self.omr._detect_exam_sheet(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        aligned = self.omr._align_to_template(corrected, t, res)
        ag = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        self.canvas.recognition_overlay.clear()
        for z in t.zones:
            if not z.grid: continue
            states = []
            radius = max(4, int(min(z.width / max(1, z.grid.cols), z.height / max(1, z.grid.rows)) / 8))
            for bx, by in z.grid.bubble_positions:
                x0, y0, x1, y1 = max(0, int(bx - radius)), max(0, int(by - radius)), min(ag.shape[1], int(bx + radius)), min(ag.shape[0], int(by + radius))
                roi = ag[y0:y1, x0:x1]
                if roi.size == 0: states.append(False); continue
                _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                states.append(float(np.count_nonzero(th)) / float(th.size) >= self.omr.fill_threshold)
            self.canvas.recognition_overlay[z.id] = states

        self.result_box.setPlainText(f"Student ID: {res.student_id or '-'}\nExam Code: {res.exam_code or '-'}\nAnswers: {', '.join([f'Q{k}:{v}' for k,v in sorted(res.answers.items())]) or '(none)'}")
        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_ok = True

    def _validate(self) -> list[str]:
        if not self.template: return ["No template loaded."]
        errs = self.template_engine.validate_template(self.template)
        for z in self.template.zones:
            if z.zone_type in BLOCK_TYPES and not z.grid:
                errs.append(f"{z.name}: grid missing.")
            if z.grid:
                exp = z.grid.question_count * max(1, len(z.grid.options))
                if len(z.grid.bubble_positions) != exp:
                    errs.append(f"{z.name}: bubble count mismatch.")
        return errs

    def save_template(self):
        if not self.template:
            return
        if not self.preview_ok:
            QMessageBox.warning(self, "Preview required", "Run preview before save."); return
        if not self.test_ok:
            QMessageBox.warning(self, "Test required", "Run test recognition before save."); return
        errs = self._validate()
        if errs:
            QMessageBox.warning(self, "Validation", "\n".join(errs)); return
        p, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if p:
            self.template.save_json(p)
