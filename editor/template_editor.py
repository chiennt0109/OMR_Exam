from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QBrush, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.template_engine import GridSpec, TemplateEngine
from models.template import AnchorPoint, Template, Zone, ZoneType


class CanvasView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class TemplateEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Editor")
        self.template_engine = TemplateEngine()
        self.scene = QGraphicsScene(self)
        self.view = CanvasView(self.scene)
        self.template: Template | None = None
        self.background_item: QGraphicsPixmapItem | None = None
        self.selected_zone_type = ZoneType.ANSWER_GRID
        self._current_rect: QGraphicsRectItem | None = None
        self._start_point = QPointF()
        self._zone_items: dict[str, QGraphicsRectItem] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Editor")
        self.addToolBar(toolbar)

        load_action = QAction("Load Blank Paper", self)
        load_action.triggered.connect(self.load_image)
        toolbar.addAction(load_action)

        save_action = QAction("Save Template JSON", self)
        save_action.triggered.connect(self.save_template)
        toolbar.addAction(save_action)

        for zone_type in ZoneType:
            action = QAction(zone_type.value, self)
            action.triggered.connect(lambda _=False, t=zone_type: self._set_zone_type(t))
            toolbar.addAction(action)

        controls = QHBoxLayout()
        gen_grid_btn = QPushButton("Auto Generate Grid")
        gen_grid_btn.clicked.connect(self.auto_generate_grid)
        dup_zone_btn = QPushButton("Duplicate Last Zone")
        dup_zone_btn.clicked.connect(self.duplicate_last_zone)
        controls.addWidget(gen_grid_btn)
        controls.addWidget(dup_zone_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(controls)

        wrapper = QWidget()
        wrapper.setLayout(layout)
        self.setCentralWidget(wrapper)

        self.view.viewport().installEventFilter(self)

    def _set_zone_type(self, zone_type: ZoneType) -> None:
        self.selected_zone_type = zone_type
        self.statusBar().showMessage(f"Zone type: {zone_type.value}")

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Blank Paper", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        pix = QPixmap(file_path)
        self.scene.clear()
        self.background_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.template = Template(
            name=Path(file_path).stem,
            image_path=file_path,
            width=pix.width(),
            height=pix.height(),
        )

    def save_template(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "No template", "Load paper image first.")
            return
        errors = self.template.validate()
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors))
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Template", "template.json", "JSON (*.json)")
        if file_path:
            self.template.save_json(file_path)

    def auto_generate_grid(self) -> None:
        if not self.template:
            return
        zones = self.template_engine.generate_auto_grid(
            base_name="Q",
            zone_type=ZoneType.ANSWER_GRID,
            x=50,
            y=50,
            cell_width=120,
            cell_height=32,
            spec=GridSpec(rows=10, cols=4, h_gap=10, v_gap=8),
            question_start=1,
        )
        for zone in zones:
            self.template.zones.append(zone)
            self._add_zone_item(zone)

    def duplicate_last_zone(self) -> None:
        if not self.template or not self.template.zones:
            return
        zone = self.template_engine.duplicate_zone(self.template.zones[-1])
        self.template.zones.append(zone)
        self._add_zone_item(zone)

    def _add_zone_item(self, zone: Zone) -> None:
        pen = QPen(Qt.red, 2)
        item = self.scene.addRect(zone.x, zone.y, zone.width, zone.height, pen)
        self._zone_items[zone.id] = item

    def eventFilter(self, source, event):
        if source is self.view.viewport() and self.template:
            if event.type() == event.MouseButtonPress and event.button() == Qt.RightButton:
                pos = self.view.mapToScene(event.position().toPoint())
                if len(self.template.anchors) < 12:
                    anchor = AnchorPoint(x=self.template_engine.snap(int(pos.x())), y=self.template_engine.snap(int(pos.y())), name=f"A{len(self.template.anchors)+1}")
                    self.template.anchors.append(anchor)
                    self.scene.addEllipse(anchor.x - 4, anchor.y - 4, 8, 8, QPen(Qt.blue, 1), QBrush(Qt.blue))
                return True
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                self._start_point = self.view.mapToScene(event.position().toPoint())
                self._current_rect = self.scene.addRect(QRectF(self._start_point, self._start_point), QPen(Qt.green, 2))
                return True
            if event.type() == event.MouseMove and self._current_rect:
                end = self.view.mapToScene(event.position().toPoint())
                rect = QRectF(self._start_point, end).normalized()
                self._current_rect.setRect(rect)
                return True
            if event.type() == event.MouseButtonRelease and self._current_rect:
                rect = self._current_rect.rect()
                zone = Zone(
                    id=str(uuid.uuid4()),
                    name=f"{self.selected_zone_type.value}_{len(self.template.zones) + 1}",
                    zone_type=self.selected_zone_type,
                    x=self.template_engine.snap(int(rect.x())),
                    y=self.template_engine.snap(int(rect.y())),
                    width=max(5, self.template_engine.snap(int(rect.width()))),
                    height=max(5, self.template_engine.snap(int(rect.height()))),
                )
                self.scene.removeItem(self._current_rect)
                self._current_rect = None
                self.template.zones.append(zone)
                self._add_zone_item(zone)
                return True
        return super().eventFilter(source, event)
