from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any


class ZoneType(str, Enum):
    ANCHOR = "ANCHOR"
    STUDENT_ID = "STUDENT_ID"
    EXAM_CODE = "EXAM_CODE"
    MCQ_BLOCK = "MCQ_BLOCK"
    TRUE_FALSE_GROUP = "TRUE_FALSE_GROUP"
    NUMERIC_GRID = "NUMERIC_GRID"


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
    bubble_positions: list[tuple[float, float]] = field(default_factory=list)  # relative coordinates


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
        for a in self.anchors:
            if not (0.0 <= a.x <= 1.0 and 0.0 <= a.y <= 1.0):
                errors.append(f"Anchor {a.name or '?'} is outside relative bounds [0,1].")
        for z in self.zones:
            if not (0.0 <= z.x <= 1.0 and 0.0 <= z.y <= 1.0 and 0.0 < z.width <= 1.0 and 0.0 < z.height <= 1.0):
                errors.append(f"Zone {z.name} has invalid relative geometry.")
        if not self.zones:
            errors.append("Template must contain at least one recognition zone.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for zone in payload["zones"]:
            zt = zone.get("zone_type")
            zone["zone_type"] = zt.value if hasattr(zt, "value") else str(zt)
        payload.setdefault("metadata", {})["coordinate_mode"] = "relative"
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Template":
        anchors = [AnchorPoint(**a) for a in data.get("anchors", [])]
        zones: list[Zone] = []
        for raw in data.get("zones", []):
            grid = BubbleGrid(**raw["grid"]) if raw.get("grid") else None
            zones.append(
                Zone(
                    id=raw["id"],
                    name=raw["name"],
                    zone_type=ZoneType(raw["zone_type"]),
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    width=float(raw["width"]),
                    height=float(raw["height"]),
                    grid=grid,
                    metadata=raw.get("metadata", {}),
                )
            )
        return cls(
            name=data["name"],
            image_path=data["image_path"],
            width=data["width"],
            height=data["height"],
            anchors=anchors,
            zones=zones,
            metadata=data.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Template":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
