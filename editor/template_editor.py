from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
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
class MCQConfig:
    total_questions: int
    choices_per_question: int
    questions_per_column: int
    column_count: int


class TemplateCanvas(QWidget):
    selection_changed = Signal(int)
    zones_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.template: Template | None = None
        self.pixmap: QPixmap | None = None
        self.zoom = 1.0

        self.add_anchor_mode = False
        self.current_zone_type = ZoneType.STUDENT_ID
        self.preview_mode = False
        self.selected_zone: int = -1

        self.recognition_overlay: dict[str, list[bool]] = {}

        self._drawing = False
        self._moving = False
        self._resizing = False
        self._resize_corner = ""
        self._start = QPoint()
        self._current_rect = QRect()
        self._move_offset = QPoint()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_template(self, template: Template, pixmap: QPixmap) -> None:
        self.template = template
        self.pixmap = pixmap
        self.zoom = 1.0
        self.selected_zone = -1
        self.preview_mode = False
        self.recognition_overlay.clear()
        self._update_canvas_size()
        self.selection_changed.emit(-1)
        self.update()

    def set_zoom(self, value: float) -> None:
        self.zoom = max(0.25, min(4.0, value))
        self._update_canvas_size()
        self.update()

    def _update_canvas_size(self) -> None:
        if not self.pixmap:
            self.resize(800, 600)
            return
        self.resize(int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))

    def _to_image(self, p: QPoint) -> QPoint:
        return QPoint(int(p.x() / self.zoom), int(p.y() / self.zoom))

    def _rel(self, x: int, y: int) -> tuple[float, float]:
        assert self.template
        return x / max(1, self.template.width), y / max(1, self.template.height)

    def _abs_zone_rect(self, zone: Zone) -> QRect:
        assert self.template
        return QRect(
            int(zone.x * self.template.width),
            int(zone.y * self.template.height),
            max(1, int(zone.width * self.template.width)),
            max(1, int(zone.height * self.template.height)),
        )

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if not self.template or not self.pixmap or e.button() != Qt.LeftButton:
            return
        pt = self._to_image(e.position().toPoint())

        if self.add_anchor_mode:
            rx, ry = self._rel(pt.x(), pt.y())
            self.template.anchors.append(AnchorPoint(x=rx, y=ry, name=f"A{len(self.template.anchors)+1}"))
            self.update()
            self.zones_changed.emit()
            return

        hit = self._hit_zone(pt)
        if hit >= 0:
            self.selected_zone = hit
            self.selection_changed.emit(hit)
            zone = self.template.zones[hit]
            corner = self._hit_corner(zone, pt)
            if corner:
                self._resizing = True
                self._resize_corner = corner
            else:
                self._moving = True
                rect = self._abs_zone_rect(zone)
                self._move_offset = QPoint(pt.x() - rect.x(), pt.y() - rect.y())
            self.update()
            return

        self.selected_zone = -1
        self.selection_changed.emit(-1)
        self._drawing = True
        self._start = pt
        self._current_rect = QRect(pt, pt)
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self.template:
            return
        pt = self._to_image(e.position().toPoint())

        if self._drawing:
            self._current_rect = QRect(self._start, pt).normalized()
            self.update()
            return

        if self.selected_zone < 0:
            return

        zone = self.template.zones[self.selected_zone]
        rect = self._abs_zone_rect(zone)

        if self._moving:
            rect.moveTo(pt.x() - self._move_offset.x(), pt.y() - self._move_offset.y())
            self._write_zone_from_abs(zone, rect)
            self.update()
            self.zones_changed.emit()
            return

        if self._resizing:
            if "left" in self._resize_corner:
                rect.setLeft(pt.x())
            if "right" in self._resize_corner:
                rect.setRight(pt.x())
            if "top" in self._resize_corner:
                rect.setTop(pt.y())
            if "bottom" in self._resize_corner:
                rect.setBottom(pt.y())
            self._write_zone_from_abs(zone, rect.normalized())
            self.update()
            self.zones_changed.emit()

    def _write_zone_from_abs(self, zone: Zone, rect: QRect) -> None:
        assert self.template
        w, h = max(1, self.template.width), max(1, self.template.height)
        zone.x = max(0.0, rect.x() / w)
        zone.y = max(0.0, rect.y() / h)
        zone.width = min(1.0, max(0.001, rect.width() / w))
        zone.height = min(1.0, max(0.001, rect.height() / h))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if not self.template or e.button() != Qt.LeftButton:
            return

        if self._drawing:
            self._drawing = False
            rect = self._current_rect.normalized()
            if rect.width() > 8 and rect.height() > 8:
                zone = Zone(
                    id=str(uuid.uuid4()),
                    name=f"{self.current_zone_type.value}_{len(self.template.zones)+1}",
                    zone_type=self.current_zone_type,
                    x=rect.x() / self.template.width,
                    y=rect.y() / self.template.height,
                    width=rect.width() / self.template.width,
                    height=rect.height() / self.template.height,
                    metadata={},
                )
                self.template.zones.append(zone)
                self.selected_zone = len(self.template.zones) - 1
                self.selection_changed.emit(self.selected_zone)
                self.zones_changed.emit()
            self._current_rect = QRect()

        self._moving = False
        self._resizing = False
        self._resize_corner = ""
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if self.template and self.selected_zone >= 0 and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_zone < len(self.template.zones):
                del self.template.zones[self.selected_zone]
            self.selected_zone = -1
            self.selection_changed.emit(-1)
            self.zones_changed.emit()
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self.pixmap:
            p.fillRect(self.rect(), Qt.lightGray)
            return

        p.drawPixmap(QRect(0, 0, int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom)), self.pixmap)
        if not self.template:
            return

        p.setPen(QPen(Qt.black, 2))
        p.setBrush(Qt.black)
        for a in self.template.anchors:
            p.drawEllipse(QPointF(a.x * self.template.width * self.zoom, a.y * self.template.height * self.zoom), 4, 4)

        p.setPen(QPen(QColor(220, 50, 50), 2))
        p.setBrush(Qt.NoBrush)
        for idx, z in enumerate(self.template.zones):
            r = self._abs_zone_rect(z)
            rz = QRectF(r.x() * self.zoom, r.y() * self.zoom, r.width() * self.zoom, r.height() * self.zoom)
            p.drawRect(rz)
            if idx == self.selected_zone and not self.preview_mode:
                self._draw_handles(p, rz)
            if self.preview_mode and z.grid:
                self._draw_grid_preview(p, z)
            if z.id in self.recognition_overlay and z.grid:
                self._draw_recognition(p, z)

        if self._drawing and self._current_rect.isValid() and not self.preview_mode:
            p.setPen(QPen(Qt.green, 2, Qt.DashLine))
            r = self._current_rect
            p.drawRect(QRectF(r.x() * self.zoom, r.y() * self.zoom, r.width() * self.zoom, r.height() * self.zoom))

    def _draw_handles(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(Qt.white)
        for pt in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))

    def _draw_grid_preview(self, painter: QPainter, zone: Zone) -> None:
        assert self.template and zone.grid
        options = zone.grid.options
        choices = max(1, len(options))
        painter.setPen(QPen(QColor(70, 70, 70), 1))

        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            x = bx * self.template.width * self.zoom
            y = by * self.template.height * self.zoom
            painter.drawEllipse(QPointF(x, y), 2.6, 2.6)

            q_idx = i // choices
            c_idx = i % choices
            q_no = zone.grid.question_start + q_idx
            if c_idx == 0:
                painter.drawText(QPointF(x - 22, y + 4), str(q_no))
            painter.drawText(QPointF(x + 4, y - 2), options[c_idx])

    def _draw_recognition(self, painter: QPainter, zone: Zone) -> None:
        assert self.template and zone.grid
        states = self.recognition_overlay[zone.id]
        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            filled = states[i] if i < len(states) else False
            painter.setPen(QPen(QColor(0, 180, 0) if filled else QColor(225, 55, 55), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(bx * self.template.width * self.zoom, by * self.template.height * self.zoom), 4, 4)

    def _hit_zone(self, point: QPoint) -> int:
        if not self.template:
            return -1
        for i in range(len(self.template.zones) - 1, -1, -1):
            if self._abs_zone_rect(self.template.zones[i]).contains(point):
                return i
        return -1

    def _hit_corner(self, zone: Zone, point: QPoint) -> str:
        r = self._abs_zone_rect(zone)
        corners = {
            "top_left": QPoint(r.left(), r.top()),
            "top_right": QPoint(r.right(), r.top()),
            "bottom_left": QPoint(r.left(), r.bottom()),
            "bottom_right": QPoint(r.right(), r.bottom()),
        }
        for name, pt in corners.items():
            if abs(point.x() - pt.x()) <= 8 and abs(point.y() - pt.y()) <= 8:
                return name
        return ""


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1580, 920)

        self.template_engine = TemplateEngine()
        self.omr_processor = OMRProcessor()
        self.template: Template | None = None
        self.preview_ok = False
        self.test_ok = False
        self._property_sync = False

        self.canvas = TemplateCanvas()
        self.canvas.selection_changed.connect(self._on_zone_selected)
        self.canvas.zones_changed.connect(self._on_zone_geometry_or_data_changed)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.canvas)

        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        a_load = QAction("Load Blank Paper", self); a_load.triggered.connect(self.load_image); toolbar.addAction(a_load)
        a_preview = QAction("Preview Template", self); a_preview.triggered.connect(self.preview_template); toolbar.addAction(a_preview)
        a_test = QAction("Test Recognition", self); a_test.triggered.connect(self.test_recognition); toolbar.addAction(a_test)
        a_save = QAction("Save Template JSON", self); a_save.triggered.connect(self.save_template); toolbar.addAction(a_save)

        self.anchor_btn = QPushButton("Add Anchor")
        self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(lambda checked: setattr(self.canvas, "add_anchor_mode", checked))
        toolbar.addWidget(self.anchor_btn)

        toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_combo = QComboBox()
        self.zone_combo.addItems([
            ZoneType.STUDENT_ID.value,
            ZoneType.EXAM_CODE.value,
            ZoneType.MCQ_BLOCK.value,
            ZoneType.TRUE_FALSE_GROUP.value,
            ZoneType.NUMERIC_GRID.value,
        ])
        self.zone_combo.currentTextChanged.connect(lambda t: setattr(self.canvas, "current_zone_type", ZoneType(t)))
        toolbar.addWidget(self.zone_combo)

        z_in = QAction("Zoom +", self); z_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15)); toolbar.addAction(z_in)
        z_out = QAction("Zoom -", self); z_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15)); toolbar.addAction(z_out)

        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMaximumHeight(150)

        self.property_panel = self._build_property_panel()

        center = QWidget()
        layout = QHBoxLayout(center)
        left = QVBoxLayout()
        left.addWidget(scroll)
        left.addWidget(self.result_box)
        layout.addLayout(left, 1)
        layout.addWidget(self.property_panel)
        self.setCentralWidget(center)

    def _build_property_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        lay = QVBoxLayout(panel)
        lay.addWidget(QLabel("MCQ Block Properties"))

        form = QFormLayout()
        self.prop_total_questions = QSpinBox(); self.prop_total_questions.setRange(1, 2000); self.prop_total_questions.setValue(40)
        self.prop_choices = QSpinBox(); self.prop_choices.setRange(2, 10); self.prop_choices.setValue(4)
        self.prop_q_per_col = QSpinBox(); self.prop_q_per_col.setRange(1, 500); self.prop_q_per_col.setValue(10)
        self.prop_col_count = QSpinBox(); self.prop_col_count.setRange(1, 20); self.prop_col_count.setValue(4)

        form.addRow("total_questions", self.prop_total_questions)
        form.addRow("choices_per_question", self.prop_choices)
        form.addRow("questions_per_column", self.prop_q_per_col)
        form.addRow("column_count", self.prop_col_count)
        lay.addLayout(form)

        self.btn_regen = QPushButton("Regenerate Grid")
        self.btn_regen.clicked.connect(self._regenerate_selected_zone)
        lay.addWidget(self.btn_regen)

        lay.addWidget(QLabel("Changes auto-regenerate preview for MCQ_BLOCK."))
        lay.addStretch(1)

        for spin in [self.prop_total_questions, self.prop_choices, self.prop_q_per_col, self.prop_col_count]:
            spin.valueChanged.connect(self._on_property_changed)

        return panel

    def _on_zone_selected(self, index: int) -> None:
        if not self.template or index < 0 or index >= len(self.template.zones):
            self._set_property_panel_enabled(False)
            return
        zone = self.template.zones[index]
        is_mcq = zone.zone_type == ZoneType.MCQ_BLOCK
        self._set_property_panel_enabled(is_mcq)
        if is_mcq:
            self._property_sync = True
            self.prop_total_questions.setValue(int(zone.metadata.get("total_questions", 40)))
            self.prop_choices.setValue(int(zone.metadata.get("choices_per_question", 4)))
            self.prop_q_per_col.setValue(int(zone.metadata.get("questions_per_column", 10)))
            self.prop_col_count.setValue(int(zone.metadata.get("column_count", 4)))
            self._property_sync = False

    def _set_property_panel_enabled(self, enabled: bool) -> None:
        for w in [self.prop_total_questions, self.prop_choices, self.prop_q_per_col, self.prop_col_count, self.btn_regen]:
            w.setEnabled(enabled)

    def _on_property_changed(self) -> None:
        if self._property_sync:
            return
        self._regenerate_selected_zone(auto=True)

    def _on_zone_geometry_or_data_changed(self) -> None:
        if self.canvas.preview_mode:
            self.preview_template()

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid", "Unable to load image")
            return

        self.template = Template(name=Path(path).stem, image_path=path, width=pix.width(), height=pix.height())
        self.canvas.set_template(self.template, pix)
        self.preview_ok = False
        self.test_ok = False
        self.result_box.clear()

    def _selected_zone(self) -> Zone | None:
        if not self.template:
            return None
        idx = self.canvas.selected_zone
        if idx < 0 or idx >= len(self.template.zones):
            return None
        return self.template.zones[idx]

    def _regenerate_selected_zone(self, auto: bool = False) -> None:
        zone = self._selected_zone()
        if not zone:
            if not auto:
                QMessageBox.warning(self, "No zone", "Select an MCQ_BLOCK zone first.")
            return
        if zone.zone_type != ZoneType.MCQ_BLOCK:
            if not auto:
                QMessageBox.warning(self, "Unsupported", "Property panel currently edits MCQ_BLOCK only.")
            return

        cfg = MCQConfig(
            total_questions=self.prop_total_questions.value(),
            choices_per_question=self.prop_choices.value(),
            questions_per_column=self.prop_q_per_col.value(),
            column_count=self.prop_col_count.value(),
        )
        zone.grid = self._build_mcq_question_layout(zone, cfg)
        zone.metadata.update(
            {
                "total_questions": cfg.total_questions,
                "choices_per_question": cfg.choices_per_question,
                "questions_per_column": cfg.questions_per_column,
                "column_count": cfg.column_count,
            }
        )
        self.canvas.update()
        if self.canvas.preview_mode:
            self.preview_template()

    def _build_mcq_question_layout(self, zone: Zone, cfg: MCQConfig) -> BubbleGrid:
        assert self.template
        total_questions = max(1, cfg.total_questions)
        choices_per_question = max(2, cfg.choices_per_question)
        questions_per_column = max(1, cfg.questions_per_column)
        column_count = max(1, cfg.column_count)

        options = [chr(65 + i) for i in range(min(choices_per_question, 26))]

        abs_x = zone.x * self.template.width
        abs_y = zone.y * self.template.height
        abs_w = zone.width * self.template.width
        abs_h = zone.height * self.template.height

        col_w = abs_w / column_count
        row_h = abs_h / questions_per_column

        bubble_positions: list[tuple[float, float]] = []

        for q in range(total_questions):
            col = q // questions_per_column
            row = q % questions_per_column
            if col >= column_count:
                break

            x0 = abs_x + col * col_w
            y0 = abs_y + row * row_h

            for c in range(len(options)):
                cx = x0 + (c + 0.5) * (col_w / len(options))
                cy = y0 + 0.5 * row_h
                bubble_positions.append((cx / self.template.width, cy / self.template.height))

        actual_questions = len(bubble_positions) // len(options)
        return BubbleGrid(
            rows=questions_per_column,
            cols=column_count,
            question_start=1,
            question_count=actual_questions,
            options=options,
            bubble_positions=bubble_positions,
        )

    def preview_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load image first")
            return

        errors = self._validate_template()
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors))
            self.preview_ok = False
            return

        self.canvas.preview_mode = True
        self.canvas.update()
        self.preview_ok = True

    def _normalize_to_200_dpi(self, path: str) -> tuple[str, str]:
        qimg = QImage(path)
        if qimg.isNull():
            return path, ""
        dpm_x = qimg.dotsPerMeterX()
        dpi = int(round(dpm_x * 0.0254)) if dpm_x > 0 else 0
        if dpi == 200:
            return path, ""
        if dpi <= 0:
            return path, "Image DPI metadata missing; expected 200 DPI."
        img = cv2.imread(path)
        if img is None:
            return path, ""
        scale = 200.0 / dpi
        resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        out = str(Path(path).with_name(f"{Path(path).stem}_200dpi.png"))
        cv2.imwrite(out, resized)
        return out, f"Input DPI={dpi}. Image normalized to 200 DPI scale."

    def _to_absolute_template(self) -> Template:
        assert self.template
        t = Template(name=self.template.name, image_path=self.template.image_path, width=self.template.width, height=self.template.height)
        t.anchors = [AnchorPoint(x=int(a.x * t.width), y=int(a.y * t.height), name=a.name) for a in self.template.anchors]
        t.zones = []
        for z in self.template.zones:
            grid = None
            if z.grid:
                grid = BubbleGrid(
                    rows=z.grid.rows,
                    cols=z.grid.cols,
                    question_start=z.grid.question_start,
                    question_count=z.grid.question_count,
                    options=z.grid.options,
                    bubble_positions=[(bx * t.width, by * t.height) for bx, by in z.grid.bubble_positions],
                )
            t.zones.append(
                Zone(
                    id=z.id,
                    name=z.name,
                    zone_type=z.zone_type,
                    x=int(z.x * t.width),
                    y=int(z.y * t.height),
                    width=int(z.width * t.width),
                    height=int(z.height * t.height),
                    grid=grid,
                    metadata=z.metadata.copy(),
                )
            )
        return t

    def test_recognition(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load image first")
            return
        if not self.preview_ok:
            QMessageBox.warning(self, "Preview required", "Run Preview Template first")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load Scanned Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return

        norm, msg = self._normalize_to_200_dpi(path)
        if msg:
            QMessageBox.warning(self, "DPI Check", msg)

        abs_template = self._to_absolute_template()
        result = self.omr_processor.process_image(norm, abs_template)

        gray = cv2.imread(norm, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            QMessageBox.warning(self, "Recognition", "Could not load scan for overlay")
            return

        corrected = self.omr_processor._detect_exam_sheet(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        aligned = self.omr_processor._align_to_template(corrected, abs_template, result)
        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        self.canvas.recognition_overlay.clear()
        for z in abs_template.zones:
            if not z.grid:
                continue
            states = []
            radius = max(4, int(min(z.width / max(1, z.grid.cols), z.height / max(1, z.grid.rows)) / 8))
            for bx, by in z.grid.bubble_positions:
                x0, y0 = max(0, int(bx - radius)), max(0, int(by - radius))
                x1, y1 = min(aligned_gray.shape[1], int(bx + radius)), min(aligned_gray.shape[0], int(by + radius))
                roi = aligned_gray[y0:y1, x0:x1]
                if roi.size == 0:
                    states.append(False)
                    continue
                _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ratio = float(np.count_nonzero(th)) / float(th.size)
                states.append(ratio >= self.omr_processor.fill_threshold)
            self.canvas.recognition_overlay[z.id] = states

        answers = ", ".join([f"Q{k}:{v}" for k, v in sorted(result.answers.items())]) or "(none)"
        self.result_box.setPlainText(
            f"Student ID: {result.student_id or '-'}\nExam Code: {result.exam_code or '-'}\nAnswers: {answers}"
        )
        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_ok = True

    def _validate_template(self) -> list[str]:
        if not self.template:
            return ["No template loaded."]
        errors = self.template_engine.validate_template(self.template)

        for z in self.template.zones:
            if z.zone_type == ZoneType.MCQ_BLOCK and not z.grid:
                errors.append(f"{z.name}: MCQ parameters not generated.")
            if not z.grid:
                continue

            expected = z.grid.question_count * max(1, len(z.grid.options))
            if len(z.grid.bubble_positions) != expected:
                errors.append(f"{z.name}: grid mismatch ({len(z.grid.bubble_positions)} vs {expected}).")

            for i, (bx, by) in enumerate(z.grid.bubble_positions):
                if not (z.x <= bx <= z.x + z.width and z.y <= by <= z.y + z.height):
                    errors.append(f"{z.name}: bubble {i} outside zone.")

        return errors

    def save_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load image first")
            return
        if not self.preview_ok:
            QMessageBox.warning(self, "Preview required", "Preview template before saving")
            return
        if not self.test_ok:
            QMessageBox.warning(self, "Test required", "Run test recognition before saving")
            return

        errors = self._validate_template()
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if path:
            self.template.save_json(path)
            self.statusBar().showMessage(f"Saved: {path}")
