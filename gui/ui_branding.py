from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QLineEdit, QPushButton, QTableWidget, QWidget

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
THEME_ROOT = Path(__file__).resolve().parents[1] / "theme"


def asset_path(*parts: str) -> Path:
    return ASSET_ROOT.joinpath(*parts)


def _icon_from_file(path: Path) -> QIcon:
    try:
        return QIcon(str(path)) if path.exists() else QIcon()
    except Exception:
        return QIcon()


@lru_cache(maxsize=512)
def icon(*parts: str) -> QIcon:
    return _icon_from_file(asset_path(*parts))


def pixmap(*parts: str) -> QPixmap:
    try:
        path = asset_path(*parts)
        return QPixmap(str(path)) if path.exists() else QPixmap()
    except Exception:
        return QPixmap()


def toolbar_icon(name: str) -> QIcon:
    return icon("icons", "toolbar", f"{name}.png")


def status_icon(name: str) -> QIcon:
    return icon("icons", "status", f"{name}.png")


def misc_icon(name: str) -> QIcon:
    return icon("icons", "misc", f"{name}.png")


def app_icon() -> QIcon:
    ico = asset_path("icons", "app_icon.ico")
    png = asset_path("icons", "app_icon.png")
    alt_ico = asset_path("icons", "app", "app_icon.ico")
    alt_png = asset_path("icons", "app", "app_icon.png")
    for p in (alt_ico, ico, alt_png, png):
        if p.exists():
            return _icon_from_file(p)
    return QIcon()


def logo_main() -> QPixmap:
    return pixmap("logos", "logo_main.png")


def logo_symbol() -> QPixmap:
    return pixmap("logos", "logo_symbol.png")


def splash_pixmap() -> QPixmap:
    return pixmap("logos", "splash.png")


def load_theme(mode: str = "light") -> str:
    qss = THEME_ROOT / ("dark.qss" if str(mode or "light").lower() == "dark" else "light.qss")
    try:
        return qss.read_text(encoding="utf-8") if qss.exists() else ""
    except Exception:
        return ""


class _LazyIconMap:
    def __init__(self, builder):
        self._builder = builder

    def __getitem__(self, key: str) -> QIcon:
        return self._builder(str(key))

    def get(self, key: str, default=None):
        try:
            icon_obj = self._builder(str(key))
            if icon_obj is None or icon_obj.isNull():
                return default
            return icon_obj
        except Exception:
            return default

    def __contains__(self, key: str) -> bool:
        ico = self.get(key)
        return ico is not None and not ico.isNull()


_TOOLBAR_ALIASES = {
    "import_": "import",
    "score": "scoring",
    "calculate": "scoring",
    "calc": "scoring",
    "browse": "open",
    "view": "preview",
    "new": "add",
    "remove": "delete",
    "trash": "delete",
    "save_as": "export",
    "api": "export",
    "pdf": "report",
    "excel": "export",
}
_STATUS_ALIASES = {"success": "ok", "done": "ok", "fail": "error", "unknown": "pending"}


def _toolbar_builder(name: str) -> QIcon:
    real_name = _TOOLBAR_ALIASES.get(str(name), str(name))
    return toolbar_icon(real_name)


def _status_builder(name: str) -> QIcon:
    real_name = _STATUS_ALIASES.get(str(name), str(name))
    return status_icon(real_name)


TOOLBAR = _LazyIconMap(_toolbar_builder)
STATUS = _LazyIconMap(_status_builder)

_KEYWORD_ICON = [
    (["nhận dạng", "scan", "xử lý ảnh"], "scan"),
    (["batch"], "batch"),
    (["tính điểm", "score", "chấm"], "scoring"),
    (["phúc tra"], "recheck"),
    (["export", "xuất", "excel"], "export"),
    (["import", "nhập", "nạp"], "import"),
    (["báo cáo", "pdf", "thống kê"], "report"),
    (["thêm", "add", "tạo mới"], "add"),
    (["sửa", "edit"], "edit"),
    (["xóa", "xoá", "delete", "bỏ chọn"], "delete"),
    (["lưu", "save", "ok"], "save"),
    (["đóng", "close", "cancel", "hủy", "huỷ"], "close"),
    (["xem", "preview"], "preview"),
    (["tìm", "lọc", "search"], "search"),
    (["mẫu"], "template"),
    (["môn"], "subject"),
    (["học sinh", "sbd"], "student"),
    (["kỳ thi", "exam"], "exam"),
    (["refresh", "làm mới", "áp mapping"], "refresh"),
    (["trợ giúp", "help"], "help"),
]


def icon_name_for_text(text: str) -> str:
    value = str(text or "").strip().casefold()
    for keys, name in _KEYWORD_ICON:
        if any(k in value for k in keys):
            return name
    return ""


def brand_button(button: QPushButton, icon_name: str | None = None) -> None:
    try:
        name = icon_name or icon_name_for_text(button.text())
        if name and button.icon().isNull():
            ico = TOOLBAR.get(name)
            if ico is not None and not ico.isNull():
                button.setIcon(ico)
        button.setMinimumHeight(max(button.minimumHeight(), 34))
        button.setIconSize(QSize(20, 20))
    except Exception:
        pass


def fit_input_widget(widget: QWidget) -> None:
    try:
        if isinstance(widget, QComboBox):
            widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            widget.setMinimumContentsLength(max(widget.minimumContentsLength(), 12))
            widget.setMinimumWidth(max(widget.minimumWidth(), 180))
            if widget.lineEdit() is not None:
                widget.lineEdit().setClearButtonEnabled(True)
        elif isinstance(widget, QLineEdit):
            widget.setMinimumWidth(max(widget.minimumWidth(), 180))
            widget.setMaximumWidth(16777215)
            widget.setClearButtonEnabled(True)
            widget.setToolTip(widget.text() or widget.placeholderText() or "")
        elif isinstance(widget, QTableWidget):
            widget.setAlternatingRowColors(True)
    except Exception:
        pass


def apply_widget_branding(root: QWidget, theme: str | None = None) -> None:
    try:
        if theme:
            qss = load_theme(theme)
            if qss:
                root.setStyleSheet(qss)
    except Exception:
        pass
    try:
        if hasattr(root, "setWindowIcon"):
            ico = app_icon()
            if not ico.isNull():
                root.setWindowIcon(ico)
    except Exception:
        pass
    try:
        for btn in root.findChildren(QPushButton):
            brand_button(btn)
        for box in root.findChildren(QDialogButtonBox):
            for btn in box.findChildren(QPushButton):
                brand_button(btn)
        for widget in root.findChildren(QWidget):
            fit_input_widget(widget)
    except Exception:
        pass
