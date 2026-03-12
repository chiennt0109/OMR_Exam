from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
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
    QToolBar,
    QVBoxLayout,
    QWidget,
)

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

        self.questions = QSpinBox()
        self.questions.setRange(1, 500)
        self.questions.setValue(40)

        self.choices = QSpinBox()
        self.choices.setRange(2, 10)
        self.choices.setValue(5)

        self.rows = QSpinBox()
        self.rows.setRange(1, 100)
        self.rows.setValue(10)

        self.cols = QSpinBox()
        self.cols.setRange(1, 100)
        self.cols.setValue(4)

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
        return GridConfig(
            questions=self.questions.value(),
            choices=self.choices.value(),
            rows=self.rows.value(),
            cols=self.cols.value(),
        )


class TemplateCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.template: Template | None = None
        self.pixmap: QPixmap | None = None
        self.zoom = 1.0

        self.add_anchor_mode = False
        self.current_zone_type = ZoneType.STUDENT_ID
        self.selected_zone: int | None = None

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
        self._update_canvas_size()
        self.update()

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
            self.template.anchors.append(AnchorPoint(x=img_pt.x(), y=img_pt.y(), name=f"A{len(self.template.anchors) + 1}"))
            self.update()
            return

        hit_zone = self._hit_zone(img_pt)
        if hit_zone is not None:
            self.selected_zone = hit_zone
            corner = self._hit_resize_corner(self.template.zones[hit_zone], img_pt)
            if corner:
                self._resizing = True
                self._resize_corner = corner
                self._start = img_pt
            else:
                self._moving = True
                zone = self.template.zones[hit_zone]
                self._move_offset = QPoint(img_pt.x() - zone.x, img_pt.y() - zone.y)
            self.update()
            return

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
            self.update()
            return

        if self._moving and self.selected_zone is not None:
            zone = self.template.zones[self.selected_zone]
            zone.x = img_pt.x() - self._move_offset.x()
            zone.y = img_pt.y() - self._move_offset.y()
            self.update()
            return

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
                    name=f"{self.current_zone_type.value}_{len(self.template.zones) + 1}",
                    zone_type=self.current_zone_type,
                    x=rect.x(),
                    y=rect.y(),
                    width=rect.width(),
                    height=rect.height(),
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

        # Anchors: black circles
        p.setPen(QPen(Qt.black, 2))
        p.setBrush(Qt.black)
        for anchor in self.template.anchors:
            center = QPointF(anchor.x * self.zoom, anchor.y * self.zoom)
            p.drawEllipse(center, 4, 4)

        # Zones: red rectangles
        zone_pen = QPen(QColor(220, 50, 50), 2)
        p.setBrush(Qt.NoBrush)
        p.setPen(zone_pen)
        for i, zone in enumerate(self.template.zones):
            rect = QRectF(zone.x * self.zoom, zone.y * self.zoom, zone.width * self.zoom, zone.height * self.zoom)
            p.drawRect(rect)

            if i == self.selected_zone:
                self._draw_resize_handles(p, rect)

            if zone.grid and zone.grid.bubble_positions:
                p.setPen(QPen(QColor(40, 40, 40), 1))
                for bx, by in zone.grid.bubble_positions:
                    p.drawEllipse(QPointF(bx * self.zoom, by * self.zoom), 2.5, 2.5)
                p.setPen(zone_pen)

        if self._drawing and self._current_rect.isValid():
            p.setPen(QPen(Qt.green, 2, Qt.DashLine))
            p.drawRect(
                QRectF(
                    self._current_rect.x() * self.zoom,
                    self._current_rect.y() * self.zoom,
                    self._current_rect.width() * self.zoom,
                    self._current_rect.height() * self.zoom,
                )
            )

    def _draw_resize_handles(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.black, 1))
        s = 8
        for pt in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.drawRect(QRectF(pt.x() - s / 2, pt.y() - s / 2, s, s))

    def _hit_zone(self, point: QPoint) -> int | None:
        if not self.template:
            return None
        for i in range(len(self.template.zones) - 1, -1, -1):
            zone = self.template.zones[i]
            if QRect(zone.x, zone.y, zone.width, zone.height).contains(point):
                return i
        return None

    def _hit_resize_corner(self, zone: Zone, point: QPoint) -> str:
        corners = {
            "top_left": QPoint(zone.x, zone.y),
            "top_right": QPoint(zone.x + zone.width, zone.y),
            "bottom_left": QPoint(zone.x, zone.y + zone.height),
            "bottom_right": QPoint(zone.x + zone.width, zone.y + zone.height),
        }
        tolerance = 8
        for name, pt in corners.items():
            if abs(point.x() - pt.x()) <= tolerance and abs(point.y() - pt.y()) <= tolerance:
                return name
        return ""

    def _setup_answer_grid(self, zone: Zone) -> None:
        dialog = GridConfigDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        cfg = dialog.values()

        rows = max(1, cfg.rows)
        cols = max(1, cfg.cols)
        choices = max(2, cfg.choices)
        question_count = cfg.questions

        options = [chr(65 + i) for i in range(min(choices, 26))]
        bubble_positions: list[tuple[float, float]] = []

        cell_w = zone.width / cols
        cell_h = zone.height / rows
        bubbles_per_question = choices
        max_questions = min(question_count, rows * cols)

        for q in range(max_questions):
            row = q // cols
            col = q % cols
            x0 = zone.x + col * cell_w
            y0 = zone.y + row * cell_h
            for i in range(bubbles_per_question):
                cx = x0 + ((i + 0.5) * (cell_w / bubbles_per_question))
                cy = y0 + (cell_h * 0.5)
                bubble_positions.append((cx, cy))

        zone.grid = BubbleGrid(
            rows=rows,
            cols=cols,
            question_start=1,
            question_count=max_questions,
            options=options,
            bubble_positions=bubble_positions,
        )


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.resize(1400, 900)

        self.template_engine = TemplateEngine()
        self.template: Template | None = None

        self.canvas = TemplateCanvas()
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.canvas)

        toolbar = QToolBar("Template Tools")
        self.addToolBar(toolbar)

        load_action = QAction("Load Blank Paper", self)
        load_action.triggered.connect(self.load_image)
        toolbar.addAction(load_action)

        save_action = QAction("Save Template JSON", self)
        save_action.triggered.connect(self.save_template)
        toolbar.addAction(save_action)

        self.anchor_btn = QPushButton("Add Anchor")
        self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(self.on_anchor_toggle)
        toolbar.addWidget(self.anchor_btn)

        toolbar.addWidget(QLabel(" Zone Type: "))
        self.zone_combo = QComboBox()
        self.zone_combo.addItems([ZoneType.STUDENT_ID.value, ZoneType.EXAM_CODE.value, ZoneType.ANSWER_GRID.value])
        self.zone_combo.currentTextChanged.connect(self.on_zone_type_change)
        toolbar.addWidget(self.zone_combo)

        zoom_in = QAction("Zoom +", self)
        zoom_in.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.15))
        toolbar.addAction(zoom_in)

        zoom_out = QAction("Zoom -", self)
        zoom_out.triggered.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.15))
        toolbar.addAction(zoom_out)

        help_label = QLabel("Left-drag: draw/move/resize zones • Delete: remove selected zone")

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(scroll)
        layout.addWidget(help_label)
        self.setCentralWidget(wrapper)

    def on_anchor_toggle(self, checked: bool) -> None:
        self.canvas.add_anchor_mode = checked
        self.statusBar().showMessage("Anchor mode active" if checked else "Zone mode active")

    def on_zone_type_change(self, text: str) -> None:
        self.canvas.current_zone_type = ZoneType(text)

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Invalid image", "Unable to load the selected image.")
            return

        self.template = Template(
            name=Path(file_path).stem,
            image_path=file_path,
            width=pixmap.width(),
            height=pixmap.height(),
        )
        self.canvas.set_template(self.template, pixmap)
        self.statusBar().showMessage(f"Loaded image: {Path(file_path).name} ({pixmap.width()}x{pixmap.height()})")

    def save_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load paper image first.")
            return

        errors = self.template_engine.validate_template(self.template)
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors))
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if not file_path:
            return

        self.template.save_json(file_path)
        self.statusBar().showMessage(f"Template saved: {file_path}")
