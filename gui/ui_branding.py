from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
THEME_ROOT = Path(__file__).resolve().parents[1] / "theme"


def asset_path(*parts: str) -> Path:
    return ASSET_ROOT.joinpath(*parts)


def _icon_from_file(path: Path) -> QIcon:
    try:
        return QIcon(str(path)) if path.exists() else QIcon()
    except Exception:
        return QIcon()


def _pixmap_from_file(path: Path) -> QPixmap:
    try:
        return QPixmap(str(path)) if path.exists() else QPixmap()
    except Exception:
        return QPixmap()


@lru_cache(maxsize=256)
def icon(*parts: str) -> QIcon:
    return _icon_from_file(asset_path(*parts))


def pixmap(*parts: str) -> QPixmap:
    # Must be called only after QApplication/QGuiApplication exists.
    return _pixmap_from_file(asset_path(*parts))


def toolbar_icon(name: str) -> QIcon:
    return icon("icons", "toolbar", f"{name}.png")


def status_icon(name: str) -> QIcon:
    return icon("icons", "status", f"{name}.png")


def misc_icon(name: str) -> QIcon:
    return icon("icons", "misc", f"{name}.png")


def app_icon() -> QIcon:
    ico = asset_path("icons", "app_icon.ico")
    png = asset_path("icons", "app_icon.png")
    return _icon_from_file(ico if ico.exists() else png)


def logo_main() -> QPixmap:
    return pixmap("logos", "logo_main.png")


def logo_symbol() -> QPixmap:
    return pixmap("logos", "logo_symbol.png")


def splash_pixmap() -> QPixmap:
    return pixmap("logos", "splash.png")


def load_theme(mode: str = "light") -> str:
    mode = (mode or "light").strip().lower()
    qss = THEME_ROOT / ("dark.qss" if mode == "dark" else "light.qss")
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
        try:
            icon_obj = self._builder(str(key))
            return icon_obj is not None and not icon_obj.isNull()
        except Exception:
            return False


def _toolbar_builder(name: str) -> QIcon:
    aliases = {
        "import": "import",
        "import_": "import",
        "score": "scoring",
        "scoring": "scoring",
        "scan": "scan",
        "batch": "batch",
        "recheck": "recheck",
        "export": "export",
        "settings": "settings",
        "report": "report",
        "edit": "edit",
        "save": "save",
        "delete": "delete",
        "refresh": "refresh",
        "search": "search",
        "home": "home",
        "subject": "subject",
        "template": "template",
        "student": "student",
        "exam": "exam",
        "close": "close",
        "help": "help",
    }
    real_name = aliases.get(name, name)
    return toolbar_icon(real_name)


def _status_builder(name: str) -> QIcon:
    aliases = {
        "ok": "ok",
        "success": "ok",
        "error": "error",
        "warning": "warning",
        "edited": "edited",
        "pending": "pending",
        "processing": "processing",
        "locked": "locked",
        "unlocked": "unlocked",
        "recognized": "recognized",
        "not_recognized": "not_recognized",
        "db_synced": "db_synced",
        "db_unsynced": "db_unsynced",
    }
    real_name = aliases.get(name, name)
    return status_icon(real_name)


TOOLBAR = _LazyIconMap(_toolbar_builder)
STATUS = _LazyIconMap(_status_builder)
