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
)

from core.omr_engine import OMRProcessor
from core.template_engine import TemplateEngine
from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


@dataclass
class GridConfig:
    questions: int
    choices: int
    rows: int
    cols: int


class GridConfigDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Answer Grid Setup")

        self.questions = QSpinBox(); self.questions.setRange(1, 500); self.questions.setValue(40)
        self.choices = QSpinBox(); self.choices.setRange(2, 10); self.choices.setValue(5)
        self.rows = QSpinBox(); self.rows.setRange(1, 100); self.rows.setValue(10)
        self.cols = QSpinBox(); self.cols.setRange(1, 100); self.cols.setValue(4)

        form = QFormLayout()
        form.addRow("Questions", self.questions)
        form.addRow("Choices", self.choices)
        form.addRow("Rows", self.rows)
        form.addRow("Columns", self.cols)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> GridConfig:
        return GridConfig(self.questions.value(), self.choices.value(), self.rows.value(), self.cols.value())


class TemplateCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.template: Template | None = None
        self.pixmap: QPixmap | None = None
        self.zoom = 1.0

        self.add_anchor_mode = False
        self.current_zone_type = ZoneType.STUDENT_ID
        self.selected_zone: int | None = None
        self.preview_mode = False

        self.recognition_overlay: dict[str, list[bool]] = {}
        self.recognition_text: dict[str, list[str]] = {}

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
        self.clear_recognition_overlay()
        self._update_canvas_size()
        self.update()

    def clear_recognition_overlay(self) -> None:
        self.recognition_overlay.clear()
        self.recognition_text.clear()

    def set_zoom(self, value: float) -> None:
        self.zoom = max(0.2, min(4.0, value))
        self._update_canvas_size()
        self.update()

    def _update_canvas_size(self) -> None:
        if not self.pixmap:
            self.resize(800, 600)
            return
        self.resize(int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))

    def _to_image(self, point: QPoint) -> QPoint:
        return QPoint(int(point.x() / self.zoom), int(point.y() / self.zoom))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.template or not self.pixmap or event.button() != Qt.LeftButton:
            return
        img_pt = self._to_image(event.position().toPoint())
        self.setFocus()

        if self.add_anchor_mode:
            self.template.anchors.append(AnchorPoint(x=img_pt.x(), y=img_pt.y(), name=f"A{len(self.template.anchors)+1}"))
            self.update(); return

        hit_zone = self._hit_zone(img_pt)
        if hit_zone is not None:
            self.selected_zone = hit_zone
            corner = self._hit_resize_corner(self.template.zones[hit_zone], img_pt)
            if corner:
                self._resizing = True
                self._resize_corner = corner
            else:
                self._moving = True
                zone = self.template.zones[hit_zone]
                self._move_offset = QPoint(img_pt.x() - zone.x, img_pt.y() - zone.y)
            self.update(); return

        self.selected_zone = None
        self._drawing = True
        self._start = img_pt
        self._current_rect = QRect(img_pt, img_pt)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.template:
            return
        img_pt = self._to_image(event.position().toPoint())

        if self._drawing:
            self._current_rect = QRect(self._start, img_pt).normalized()
            self.update(); return

        if self._moving and self.selected_zone is not None:
            zone = self.template.zones[self.selected_zone]
            zone.x = img_pt.x() - self._move_offset.x()
            zone.y = img_pt.y() - self._move_offset.y()
            self.update(); return

        if self._resizing and self.selected_zone is not None:
            zone = self.template.zones[self.selected_zone]
            rect = QRect(zone.x, zone.y, zone.width, zone.height)
            if "left" in self._resize_corner:
                rect.setLeft(img_pt.x())
            if "right" in self._resize_corner:
                rect.setRight(img_pt.x())
            if "top" in self._resize_corner:
                rect.setTop(img_pt.y())
            if "bottom" in self._resize_corner:
                rect.setBottom(img_pt.y())
            rect = rect.normalized()
            zone.x, zone.y, zone.width, zone.height = rect.x(), rect.y(), max(4, rect.width()), max(4, rect.height())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.template or event.button() != Qt.LeftButton:
            return
        if self._drawing:
            self._drawing = False
            rect = self._current_rect.normalized()
            if rect.width() > 6 and rect.height() > 6:
                zone = Zone(
                    id=str(uuid.uuid4()),
                    name=f"{self.current_zone_type.value}_{len(self.template.zones)+1}",
                    zone_type=self.current_zone_type,
                    x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height(),
                )
                if zone.zone_type == ZoneType.ANSWER_GRID:
                    self._setup_answer_grid(zone)
                self.template.zones.append(zone)
                self.selected_zone = len(self.template.zones) - 1
            self._current_rect = QRect()

        self._moving = False
        self._resizing = False
        self._resize_corner = ""
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self.template:
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_zone is not None:
            if 0 <= self.selected_zone < len(self.template.zones):
                del self.template.zones[self.selected_zone]
                self.selected_zone = None
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        if self.pixmap:
            target = QRect(0, 0, int(self.pixmap.width() * self.zoom), int(self.pixmap.height() * self.zoom))
            p.drawPixmap(target, self.pixmap)
        else:
            p.fillRect(self.rect(), Qt.lightGray)
            return

        if not self.template:
            return

        p.setPen(QPen(Qt.black, 2))
        p.setBrush(Qt.black)
        for anchor in self.template.anchors:
            p.drawEllipse(QPointF(anchor.x * self.zoom, anchor.y * self.zoom), 4, 4)

        zone_pen = QPen(QColor(220, 50, 50), 2)
        p.setBrush(Qt.NoBrush)
        p.setPen(zone_pen)
        for i, zone in enumerate(self.template.zones):
            rect = QRectF(zone.x * self.zoom, zone.y * self.zoom, zone.width * self.zoom, zone.height * self.zoom)
            p.drawRect(rect)
            if i == self.selected_zone and not self.preview_mode:
                self._draw_resize_handles(p, rect)

            if self.preview_mode and zone.grid:
                self._draw_grid_preview(p, zone)

            if zone.id in self.recognition_overlay and zone.grid:
                self._draw_recognition_overlay(p, zone)

        if self._drawing and self._current_rect.isValid() and not self.preview_mode:
            p.setPen(QPen(Qt.green, 2, Qt.DashLine))
            p.drawRect(QRectF(self._current_rect.x() * self.zoom, self._current_rect.y() * self.zoom,
                              self._current_rect.width() * self.zoom, self._current_rect.height() * self.zoom))

    def _draw_grid_preview(self, painter: QPainter, zone: Zone) -> None:
        assert zone.grid
        options = zone.grid.options
        choices = max(1, len(options))
        questions = max(0, zone.grid.question_count)

        painter.setPen(QPen(QColor(90, 90, 90), 1))
        for idx, (bx, by) in enumerate(zone.grid.bubble_positions):
            painter.drawEllipse(QPointF(bx * self.zoom, by * self.zoom), 2.2, 2.2)
            q = idx // choices
            c = idx % choices
            if q < questions:
                if c == 0:
                    painter.drawText(QPointF((bx - 22) * self.zoom, (by + 4) * self.zoom), str(zone.grid.question_start + q))
                painter.drawText(QPointF((bx + 4) * self.zoom, (by - 4) * self.zoom), options[c])

    def _draw_recognition_overlay(self, painter: QPainter, zone: Zone) -> None:
        assert zone.grid
        states = self.recognition_overlay.get(zone.id, [])
        for i, (bx, by) in enumerate(zone.grid.bubble_positions):
            filled = states[i] if i < len(states) else False
            color = QColor(0, 180, 0) if filled else QColor(220, 60, 60)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 1))
            painter.drawEllipse(QPointF(bx * self.zoom, by * self.zoom), 4, 4)

    def _draw_resize_handles(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.black, 1))
        for pt in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))

    def _hit_zone(self, point: QPoint) -> int | None:
        if not self.template:
            return None
        for i in range(len(self.template.zones) - 1, -1, -1):
            z = self.template.zones[i]
            if QRect(z.x, z.y, z.width, z.height).contains(point):
                return i
        return None

    def _hit_resize_corner(self, zone: Zone, point: QPoint) -> str:
        corners = {
            "top_left": QPoint(zone.x, zone.y),
            "top_right": QPoint(zone.x + zone.width, zone.y),
            "bottom_left": QPoint(zone.x, zone.y + zone.height),
            "bottom_right": QPoint(zone.x + zone.width, zone.y + zone.height),
        }
        for name, pt in corners.items():
            if abs(point.x() - pt.x()) <= 8 and abs(point.y() - pt.y()) <= 8:
                return name
        return ""

    def _setup_answer_grid(self, zone: Zone) -> None:
        dialog = GridConfigDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        cfg = dialog.values()

        options = [chr(65 + i) for i in range(min(cfg.choices, 26))]
        rows, cols = max(1, cfg.rows), max(1, cfg.cols)
        questions = min(cfg.questions, rows * cols)

        cell_w = zone.width / cols
        cell_h = zone.height / rows
        bubble_positions: list[tuple[float, float]] = []

        for q in range(questions):
            row, col = divmod(q, cols)
            x0 = zone.x + col * cell_w
            y0 = zone.y + row * cell_h
            for i in range(len(options)):
                cx = x0 + ((i + 0.5) * (cell_w / len(options)))
                cy = y0 + cell_h * 0.5
                bubble_positions.append((cx, cy))

        zone.grid = BubbleGrid(
            rows=rows,
            cols=cols,
            question_start=1,
            question_count=questions,
            options=options,
            bubble_positions=bubble_positions,
        )


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1500, 920)

        self.template_engine = TemplateEngine()
        self.omr_processor = OMRProcessor()
        self.template: Template | None = None
        self.preview_verified = False
        self.test_verified = False

        self.canvas = TemplateCanvas()
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.canvas)

        toolbar = QToolBar("Template Tools")
        self.addToolBar(toolbar)

        load_action = QAction("Load Blank Paper", self)
        load_action.triggered.connect(self.load_image)
        toolbar.addAction(load_action)

        self.preview_action = QAction("Preview Template", self)
        self.preview_action.triggered.connect(self.preview_template)
        toolbar.addAction(self.preview_action)

        self.test_action = QAction("Test Recognition", self)
        self.test_action.triggered.connect(self.test_recognition)
        toolbar.addAction(self.test_action)

        self.save_action = QAction("Save Template JSON", self)
        self.save_action.triggered.connect(self.save_template)
        toolbar.addAction(self.save_action)

        self.anchor_btn = QPushButton("Add Anchor")
        self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(self.on_anchor_toggle)
        toolbar.addWidget(self.anchor_btn)

        toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_combo = QComboBox()
        self.zone_combo.addItems([ZoneType.STUDENT_ID.value, ZoneType.EXAM_CODE.value, ZoneType.ANSWER_GRID.value])
        self.zone_combo.currentTextChanged.connect(self.on_zone_type_change)
        toolbar.addWidget(self.zone_combo)

        zoom_in = QAction("Zoom +", self); zoom_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15))
        zoom_out = QAction("Zoom -", self); zoom_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15))
        toolbar.addAction(zoom_in); toolbar.addAction(zoom_out)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setMaximumHeight(150)

        help_label = QLabel("Draw zones with drag. Move/resize selected zones. Delete key removes selected zone.")

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(scroll)
        layout.addWidget(help_label)
        layout.addWidget(self.result_box)
        self.setCentralWidget(wrapper)

    def on_anchor_toggle(self, checked: bool) -> None:
        self.canvas.add_anchor_mode = checked
        self.canvas.preview_mode = False
        self.preview_verified = False
        self.statusBar().showMessage("Anchor mode active" if checked else "Zone mode active")

    def on_zone_type_change(self, text: str) -> None:
        self.canvas.current_zone_type = ZoneType(text)

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not file_path:
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Invalid image", "Unable to load selected image.")
            return
        self.template = Template(name=Path(file_path).stem, image_path=file_path, width=pixmap.width(), height=pixmap.height())
        self.canvas.set_template(self.template, pixmap)
        self.preview_verified = False
        self.test_verified = False
        self.result_box.clear()
        self.statusBar().showMessage(f"Loaded {Path(file_path).name}")

    def preview_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load paper image first.")
            return
        errors = self._validate_template()
        if errors:
            QMessageBox.warning(self, "Template Validation Failed", "\n".join(errors))
            self.preview_verified = False
            return
        self.canvas.preview_mode = True
        self.canvas.update()
        self.preview_verified = True
        self.statusBar().showMessage("Preview mode enabled.")

    def test_recognition(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load paper image first.")
            return
        if not self.preview_verified:
            QMessageBox.warning(self, "Preview required", "Run Preview Template before Test Recognition.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Scanned Sheet", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not file_path:
            return

        norm_path, dpi_message = self._normalize_to_200_dpi(file_path)
        if dpi_message:
            QMessageBox.warning(self, "DPI Warning", dpi_message)

        result = self.omr_processor.process_image(norm_path, self.template)

        src = cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE)
        if src is None:
            QMessageBox.warning(self, "Scan error", "Failed to load scan image for overlay analysis.")
            return
        corrected = self.omr_processor._detect_exam_sheet(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR))
        aligned = self.omr_processor._align_to_template(corrected, self.template, result)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        self.canvas.clear_recognition_overlay()
        lines = [f"Student ID: {result.student_id or '-'}", f"Exam Code: {result.exam_code or '-'}"]

        for zone in self.template.zones:
            if zone.zone_type != ZoneType.ANSWER_GRID or not zone.grid:
                continue
            states = self._analyze_zone_bubbles(gray, zone)
            self.canvas.recognition_overlay[zone.id] = states

        answers_text = ", ".join([f"Q{k}:{v}" for k, v in sorted(result.answers.items())]) or "(none)"
        lines.append(f"Answers: {answers_text}")
        if result.issues:
            lines.append("Issues: " + "; ".join([i.code for i in result.issues]))

        self.result_box.setPlainText("\n".join(lines))
        self.canvas.preview_mode = True
        self.canvas.update()
        self.test_verified = True
        self.statusBar().showMessage("Test recognition complete.")

    def _analyze_zone_bubbles(self, gray: np.ndarray, zone: Zone) -> list[bool]:
        assert zone.grid
        states: list[bool] = []
        radius = max(4, int(min(zone.width / max(1, zone.grid.cols), zone.height / max(1, zone.grid.rows)) / 8))
        for bx, by in zone.grid.bubble_positions:
            x0 = max(0, int(bx - radius)); y0 = max(0, int(by - radius))
            x1 = min(gray.shape[1], int(bx + radius)); y1 = min(gray.shape[0], int(by + radius))
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                states.append(False)
                continue
            _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fill_ratio = float(np.count_nonzero(th)) / float(th.size)
            states.append(fill_ratio >= self.omr_processor.fill_threshold)
        return states

    def _normalize_to_200_dpi(self, image_path: str) -> tuple[str, str]:
        image = QImage(image_path)
        if image.isNull():
            return image_path, ""

        dpm_x = image.dotsPerMeterX()
        dpi = int(round(dpm_x * 0.0254)) if dpm_x > 0 else 0
        if dpi <= 0:
            return image_path, "Image DPI metadata is missing; proceeding with original resolution (expected 200 DPI)."
        if dpi == 200:
            return image_path, ""

        scale = 200.0 / float(dpi)
        src = cv2.imread(image_path)
        if src is None:
            return image_path, ""
        norm = cv2.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        out_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_200dpi.png"))
        cv2.imwrite(out_path, norm)
        return out_path, f"Scan DPI is {dpi}, expected 200. Image was normalized to 200 DPI scale."

    def _validate_template(self) -> list[str]:
        if not self.template:
            return ["No template loaded."]

        errors = self.template_engine.validate_template(self.template)
        if not self.template.anchors:
            errors.append("anchors detected: no anchors defined.")

        for zone in self.template.zones:
            if zone.zone_type != ZoneType.ANSWER_GRID or not zone.grid:
                continue

            # Grid alignment checks
            choices = len(zone.grid.options)
            expected = zone.grid.question_count * choices
            if expected != len(zone.grid.bubble_positions):
                errors.append(f"grid alignment failed in {zone.name}: expected {expected} bubbles, got {len(zone.grid.bubble_positions)}.")
            if zone.grid.question_count > zone.grid.rows * zone.grid.cols:
                errors.append(f"grid alignment failed in {zone.name}: question_count exceeds rows*cols.")

            # Bubble outside zone + overlap checks
            radius = max(4, int(min(zone.width / max(1, zone.grid.cols), zone.height / max(1, zone.grid.rows)) / 8))
            for i, (bx, by) in enumerate(zone.grid.bubble_positions):
                if not (zone.x <= bx <= zone.x + zone.width and zone.y <= by <= zone.y + zone.height):
                    errors.append(f"bubble outside zone in {zone.name} at index {i}.")

            for i in range(len(zone.grid.bubble_positions)):
                x1, y1 = zone.grid.bubble_positions[i]
                for j in range(i + 1, len(zone.grid.bubble_positions)):
                    x2, y2 = zone.grid.bubble_positions[j]
                    if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 < radius * 1.5:
                        errors.append(f"bubble overlap in {zone.name} between indices {i} and {j}.")
                        break

        return errors

    def save_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load paper image first.")
            return

        if not self.preview_verified:
            QMessageBox.warning(self, "Preview required", "Run Preview Template successfully before saving.")
            return
        if not self.test_verified:
            QMessageBox.warning(self, "Test required", "Run Test Recognition successfully before saving.")
            return

        errors = self._validate_template()
        if errors:
            QMessageBox.warning(self, "Template Validation Failed", "\n".join(errors))
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if not file_path:
            return
        self.template.save_json(file_path)
        self.statusBar().showMessage(f"Template saved: {file_path}")
