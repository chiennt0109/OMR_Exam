from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any


class ZoneType(str, Enum):
    ANCHOR = "ANCHOR"
    STUDENT_ID_BLOCK = "STUDENT_ID_BLOCK"
    EXAM_CODE_BLOCK = "EXAM_CODE_BLOCK"
    MCQ_BLOCK = "MCQ_BLOCK"
    TRUE_FALSE_BLOCK = "TRUE_FALSE_BLOCK"
    NUMERIC_BLOCK = "NUMERIC_BLOCK"


@dataclass
class AnchorPoint:
    x: float
    y: float
    name: str = ""


@dataclass
class BubbleGrid:
    rows: int
    cols: int
    question_start: int
    question_count: int = 0
    options: list[str] = field(default_factory=list)
    bubble_positions: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Zone:
    id: str
    name: str
    zone_type: ZoneType
    x: float
    y: float
    width: float
    height: float
    grid: BubbleGrid | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
    name: str
    image_path: str
    width: int
    height: int
    anchors: list[AnchorPoint] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.anchors:
            errors.append("Template must contain at least one anchor point.")
        if not self.zones:
            errors.append("Template must contain at least one recognition zone.")
        for z in self.zones:
            if not (0.0 <= z.x <= 1.0 and 0.0 <= z.y <= 1.0 and 0.0 < z.width <= 1.0 and 0.0 < z.height <= 1.0):
                errors.append(f"Zone {z.name} has invalid relative geometry.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for zone in payload["zones"]:
            zt = zone.get("zone_type")
            zone["zone_type"] = zt.value if hasattr(zt, "value") else str(zt)
        md = payload.setdefault("metadata", {})
        md["coordinate_mode"] = "relative"
        md["template_width"] = int(self.width)
        md["template_height"] = int(self.height)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Template":
        legacy = {
            "TRUE_FALSE_GROUP": "TRUE_FALSE_BLOCK",
            "NUMERIC_GRID": "NUMERIC_BLOCK",
            "STUDENT_ID": "STUDENT_ID_BLOCK",
            "EXAM_CODE": "EXAM_CODE_BLOCK",
            "ID_BLOCK": "STUDENT_ID_BLOCK",
        }
        metadata = data.get("metadata", {}) or {}
        width = int(data.get("width", metadata.get("template_width", 1)) or 1)
        height = int(data.get("height", metadata.get("template_height", 1)) or 1)
        metadata.setdefault("template_width", int(width))
        metadata.setdefault("template_height", int(height))
        coordinate_mode = str(metadata.get("coordinate_mode", "")).lower()

        def _is_relative(v: float) -> bool:
            return 0.0 <= float(v) <= 1.0

        def _to_relative_x(x: float) -> float:
            xv = float(x)
            if coordinate_mode == "absolute":
                return xv / width
            if coordinate_mode == "relative":
                return xv
            return xv if _is_relative(xv) else (xv / width)

        def _to_relative_y(y: float) -> float:
            yv = float(y)
            if coordinate_mode == "absolute":
                return yv / height
            if coordinate_mode == "relative":
                return yv
            return yv if _is_relative(yv) else (yv / height)

        anchors = [
            AnchorPoint(
                x=_to_relative_x(a.get("x", 0.0)),
                y=_to_relative_y(a.get("y", 0.0)),
                name=str(a.get("name", "")),
            )
            for a in data.get("anchors", [])
        ]
        zones: list[Zone] = []
        for raw in data.get("zones", []):
            grid = None
            if raw.get("grid"):
                gr = dict(raw["grid"])
                legacy_positions = gr.get("bubble_positions", [])
                converted_positions: list[tuple[float, float]] = []
                for pos in legacy_positions:
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        continue
                    bx, by = float(pos[0]), float(pos[1])
                    converted_positions.append((_to_relative_x(bx), _to_relative_y(by)))
                gr["bubble_positions"] = converted_positions
                grid = BubbleGrid(**gr)
            zt_raw = legacy.get(raw["zone_type"], raw["zone_type"])
            zones.append(
                Zone(
                    id=raw.get("id", ""),
                    name=raw.get("name", ""),
                    zone_type=ZoneType(zt_raw),
                    x=_to_relative_x(raw.get("x", 0.0)),
                    y=_to_relative_y(raw.get("y", 0.0)),
                    width=max(1e-6, _to_relative_x(raw.get("width", 0.0))),
                    height=max(1e-6, _to_relative_y(raw.get("height", 0.0))),
                    grid=grid,
                    metadata=raw.get("metadata", {}),
                )
            )
        return cls(
            name=data["name"],
            image_path=data["image_path"],
            width=width,
            height=height,
            anchors=anchors,
            zones=zones,
            metadata=metadata,
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Template":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
