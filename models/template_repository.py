from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from models.template import Template


@dataclass
class TemplateRepository:
    templates: dict[str, str] = field(default_factory=dict)

    def register(self, template_path: str, display_name: str | None = None) -> str:
        path = str(template_path or "").strip()
        if not path:
            return ""
        name = (display_name or "").strip()
        if not name:
            p = Path(path)
            name = p.stem
            if p.exists():
                try:
                    tpl = Template.load_json(p)
                    name = str(tpl.name or name).strip() or name
                except Exception:
                    name = p.stem
        self.templates[name] = path
        return name

    def list_templates(self) -> list[tuple[str, str]]:
        return sorted(self.templates.items(), key=lambda item: item[0].lower())

    def get_path(self, display_name: str) -> str:
        return str(self.templates.get(display_name, ""))

    def to_dict(self) -> dict[str, Any]:
        return {"templates": dict(self.templates)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateRepository":
        raw = data.get("templates", {})
        templates = {str(k): str(v) for k, v in raw.items() if str(k).strip() and str(v).strip()}
        return cls(templates=templates)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TemplateRepository":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
