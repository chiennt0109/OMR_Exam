from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
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
)

from core.omr_engine import OMRProcessor
from core.template_engine import TemplateEngine
from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


@dataclass
class BlockWizardConfig:
    questions: int
    choices: int
    columns: int
    sub_questions: int
    digits: int


class BlockWizardDialog(QDialog):
    def __init__(self, zone_type: ZoneType, parent: QWidget | None = None):
        super().__init__(parent)
        self.zone_type = zone_type
        self.setWindowTitle(f"Grid Wizard - {zone_type.value}")

        self.questions = QSpinBox(); self.questions.setRange(1, 1000); self.questions.setValue(40)
        self.choices = QSpinBox(); self.choices.setRange(2, 10); self.choices.setValue(5)
        self.columns = QSpinBox(); self.columns.setRange(1, 20); self.columns.setValue(4)
        self.sub_questions = QSpinBox(); self.sub_questions.setRange(1, 10); self.sub_questions.setValue(1)
        self.digits = QSpinBox(); self.digits.setRange(1, 20); self.digits.setValue(8)

        form = QFormLayout()
        form.addRow("Questions", self.questions)

        if zone_type == ZoneType.MCQ_BLOCK:
            form.addRow("Choices", self.choices)
            form.addRow("Columns", self.columns)
        elif zone_type == ZoneType.TRUE_FALSE_GROUP:
            form.addRow("Sub Questions", self.sub_questions)
            form.addRow("Columns", self.columns)
        elif zone_type == ZoneType.NUMERIC_GRID:
            form.addRow("Digits", self.digits)
            form.addRow("Rows", QLabel("10 (fixed)"))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> BlockWizardConfig:
        return BlockWizardConfig(
            questions=self.questions.value(),
            choices=self.choices.value(),
            columns=self.columns.value(),
            sub_questions=self.sub_questions.value(),
            digits=self.digits.value(),
        )


class TemplateCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.template: Template | None = None
        self.pixmap: QPixmap | None = None
        self.zoom = 1.0

        self.add_anchor_mode = False
        self.current_zone_type = ZoneType.STUDENT_ID
        self.preview_mode = False
        self.selected_zone: int | None = None

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
        self.selected_zone = None
        self.preview_mode = False
        self.recognition_overlay.clear()
        self._update_canvas_size()
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
            self.update(); return

        hit = self._hit_zone(pt)
        if hit is not None:
            self.selected_zone = hit
            zone = self.template.zones[hit]
            corner = self._hit_corner(zone, pt)
            if corner:
                self._resizing = True
                self._resize_corner = corner
            else:
                self._moving = True
                rect = self._abs_zone_rect(zone)
                self._move_offset = QPoint(pt.x() - rect.x(), pt.y() - rect.y())
            self.update(); return

        self.selected_zone = None
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
            self.update(); return

        if self.selected_zone is None:
            return
        zone = self.template.zones[self.selected_zone]
        rect = self._abs_zone_rect(zone)

        if self._moving:
            rect.moveTo(pt.x() - self._move_offset.x(), pt.y() - self._move_offset.y())
            self._write_zone_from_abs(zone, rect)
            self.update(); return

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

    def _write_zone_from_abs(self, zone: Zone, rect: QRect) -> None:
        assert self.template
        w = max(1, self.template.width)
        h = max(1, self.template.height)
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
                )
                self.template.zones.append(zone)
                self.selected_zone = len(self.template.zones) - 1
            self._current_rect = QRect()

        self._moving = False
        self._resizing = False
        self._resize_corner = ""
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if self.template and self.selected_zone is not None and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected_zone < len(self.template.zones):
                del self.template.zones[self.selected_zone]
            self.selected_zone = None
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

        p.setPen(QPen(Qt.black, 2)); p.setBrush(Qt.black)
        for a in self.template.anchors:
            p.drawEllipse(QPointF(a.x * self.template.width * self.zoom, a.y * self.template.height * self.zoom), 4, 4)

        p.setPen(QPen(QColor(220, 50, 50), 2)); p.setBrush(Qt.NoBrush)
        for i, z in enumerate(self.template.zones):
            r = self._abs_zone_rect(z)
            rz = QRectF(r.x() * self.zoom, r.y() * self.zoom, r.width() * self.zoom, r.height() * self.zoom)
            p.drawRect(rz)
            if i == self.selected_zone and not self.preview_mode:
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
        painter.setPen(QPen(Qt.black, 1)); painter.setBrush(Qt.white)
        for pt in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.drawRect(QRectF(pt.x()-4, pt.y()-4, 8, 8))

    def _draw_grid_preview(self, painter: QPainter, zone: Zone) -> None:
        assert self.template and zone.grid
        options = zone.grid.options
        choices = max(1, len(options))
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            x = bx * self.template.width * self.zoom
            y = by * self.template.height * self.zoom
            painter.drawEllipse(QPointF(x, y), 2.4, 2.4)
            q_idx = i // choices
            c_idx = i % choices
            if c_idx == 0:
                painter.drawText(QPointF(x - 22, y + 4), str(zone.grid.question_start + q_idx))
            painter.drawText(QPointF(x + 4, y - 2), options[c_idx])

    def _draw_recognition(self, painter: QPainter, zone: Zone) -> None:
        assert self.template and zone.grid
        states = self.recognition_overlay[zone.id]
        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            filled = states[i] if i < len(states) else False
            painter.setPen(QPen(QColor(0, 190, 0) if filled else QColor(230, 60, 60), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(bx * self.template.width * self.zoom, by * self.template.height * self.zoom), 4, 4)

    def _hit_zone(self, point: QPoint) -> int | None:
        if not self.template:
            return None
        for i in range(len(self.template.zones)-1, -1, -1):
            if self._abs_zone_rect(self.template.zones[i]).contains(point):
                return i
        return None

    def _hit_corner(self, zone: Zone, point: QPoint) -> str:
        r = self._abs_zone_rect(zone)
        c = {
            "top_left": QPoint(r.left(), r.top()),
            "top_right": QPoint(r.right(), r.top()),
            "bottom_left": QPoint(r.left(), r.bottom()),
            "bottom_right": QPoint(r.right(), r.bottom()),
        }
        for name, pt in c.items():
            if abs(point.x() - pt.x()) <= 8 and abs(point.y() - pt.y()) <= 8:
                return name
        return ""


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1500, 920)

        self.template_engine = TemplateEngine()
        self.omr_processor = OMRProcessor()
        self.template: Template | None = None
        self.preview_ok = False
        self.test_ok = False

        self.canvas = TemplateCanvas()
        scroll = QScrollArea(); scroll.setWidgetResizable(False); scroll.setWidget(self.canvas)

        toolbar = QToolBar("Tools"); self.addToolBar(toolbar)
        a_load = QAction("Load Blank Paper", self); a_load.triggered.connect(self.load_image); toolbar.addAction(a_load)
        a_wizard = QAction("Generate Block Grid", self); a_wizard.triggered.connect(self.generate_grid_for_selected_zone); toolbar.addAction(a_wizard)
        a_preview = QAction("Preview Template", self); a_preview.triggered.connect(self.preview_template); toolbar.addAction(a_preview)
        a_test = QAction("Test Recognition", self); a_test.triggered.connect(self.test_recognition); toolbar.addAction(a_test)
        a_save = QAction("Save Template JSON", self); a_save.triggered.connect(self.save_template); toolbar.addAction(a_save)

        self.anchor_btn = QPushButton("Add Anchor"); self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(self.on_anchor_toggle); toolbar.addWidget(self.anchor_btn)

        toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_combo = QComboBox()
        self.zone_combo.addItems([
            ZoneType.STUDENT_ID.value,
            ZoneType.EXAM_CODE.value,
            ZoneType.MCQ_BLOCK.value,
            ZoneType.TRUE_FALSE_GROUP.value,
            ZoneType.NUMERIC_GRID.value,
        ])
        self.zone_combo.currentTextChanged.connect(lambda text: setattr(self.canvas, "current_zone_type", ZoneType(text)))
        toolbar.addWidget(self.zone_combo)

        z_in = QAction("Zoom +", self); z_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15)); toolbar.addAction(z_in)
        z_out = QAction("Zoom -", self); z_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15)); toolbar.addAction(z_out)

        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setMaximumHeight(140)

        wrapper = QWidget(); layout = QVBoxLayout(wrapper)
        layout.addWidget(scroll)
        layout.addWidget(QLabel("Draw zones by drag. Select a block zone then click Generate Block Grid."))
        layout.addWidget(self.result_box)
        self.setCentralWidget(wrapper)

    def on_anchor_toggle(self, checked: bool) -> None:
        self.canvas.add_anchor_mode = checked

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
        self.preview_ok = False; self.test_ok = False; self.result_box.clear()

    def generate_grid_for_selected_zone(self) -> None:
        if not self.template or self.canvas.selected_zone is None:
            QMessageBox.warning(self, "No zone", "Select MCQ_BLOCK / TRUE_FALSE_GROUP / NUMERIC_GRID zone first.")
            return
        zone = self.template.zones[self.canvas.selected_zone]
        if zone.zone_type not in (ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_GROUP, ZoneType.NUMERIC_GRID):
            QMessageBox.warning(self, "Unsupported", "Grid wizard applies only to block zone types.")
            return

        dlg = BlockWizardDialog(zone.zone_type, self)
        if dlg.exec() != QDialog.Accepted:
            return
        cfg = dlg.values()
        zone.grid = self._build_grid(zone, cfg)
        self.canvas.update()

    def _build_grid(self, zone: Zone, cfg: BlockWizardConfig) -> BubbleGrid:
        if zone.zone_type == ZoneType.MCQ_BLOCK:
            options = [chr(65 + i) for i in range(min(cfg.choices, 26))]
            q_count = cfg.questions
            cols = max(1, cfg.columns)
            rows = int(np.ceil(q_count / cols))
            return self._layout_grid(zone, q_count, options, rows, cols, {"columns": cols, "questions": q_count, "choices": len(options)})

        if zone.zone_type == ZoneType.TRUE_FALSE_GROUP:
            options = ["T", "F"]
            q_count = cfg.questions * max(1, cfg.sub_questions)
            cols = max(1, cfg.columns)
            rows = int(np.ceil(q_count / cols))
            return self._layout_grid(zone, q_count, options, rows, cols, {"questions": cfg.questions, "sub_questions": cfg.sub_questions})

        # NUMERIC_GRID
        options = [str(i) for i in range(10)]
        q_count = cfg.questions * max(1, cfg.digits)
        rows = 10
        cols = max(1, cfg.questions * cfg.digits)
        return self._layout_grid(zone, q_count, options, rows, cols, {"questions": cfg.questions, "digits": cfg.digits, "rows": 10})

    def _layout_grid(self, zone: Zone, question_count: int, options: list[str], rows: int, cols: int, meta: dict) -> BubbleGrid:
        assert self.template
        abs_x = zone.x * self.template.width
        abs_y = zone.y * self.template.height
        abs_w = zone.width * self.template.width
        abs_h = zone.height * self.template.height

        cell_w = abs_w / max(1, cols)
        cell_h = abs_h / max(1, rows)
        bubbles: list[tuple[float, float]] = []

        max_q = min(question_count, rows * cols)
        for q in range(max_q):
            r = q // cols
            c = q % cols
            x0 = abs_x + c * cell_w
            y0 = abs_y + r * cell_h
            for i in range(len(options)):
                cx = x0 + (i + 0.5) * (cell_w / len(options))
                cy = y0 + cell_h * 0.5
                bubbles.append((cx / self.template.width, cy / self.template.height))

        zone.metadata.update(meta)
        return BubbleGrid(rows=rows, cols=cols, question_start=1, question_count=max_q, options=options, bubble_positions=bubbles)

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
        zones: list[Zone] = []
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
            zones.append(Zone(
                id=z.id, name=z.name, zone_type=z.zone_type,
                x=int(z.x * t.width), y=int(z.y * t.height), width=int(z.width * t.width), height=int(z.height * t.height),
                grid=grid, metadata=z.metadata.copy()
            ))
        t.zones = zones
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
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        corrected = self.omr_processor._detect_exam_sheet(color)
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
                    states.append(False); continue
                _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ratio = float(np.count_nonzero(th)) / float(th.size)
                states.append(ratio >= self.omr_processor.fill_threshold)
            self.canvas.recognition_overlay[z.id] = states

        answers = ", ".join([f"Q{k}:{v}" for k, v in sorted(result.answers.items())]) or "(none)"
        self.result_box.setPlainText(f"Student ID: {result.student_id or '-'}\nExam Code: {result.exam_code or '-'}\nAnswers: {answers}")
        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_ok = True

    def _validate_template(self) -> list[str]:
        if not self.template:
            return ["No template loaded."]
        errors = self.template_engine.validate_template(self.template)

        for z in self.template.zones:
            if z.zone_type in (ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_GROUP, ZoneType.NUMERIC_GRID) and not z.grid:
                errors.append(f"{z.name}: grid wizard has not been run.")
            if not z.grid:
                continue
            expected = z.grid.question_count * max(1, len(z.grid.options))
            if len(z.grid.bubble_positions) != expected:
                errors.append(f"{z.name}: grid alignment mismatch ({len(z.grid.bubble_positions)} vs {expected}).")
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
