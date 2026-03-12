from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any


class ZoneType(str, Enum):
    STUDENT_ID = "STUDENT_ID"
    EXAM_CODE = "EXAM_CODE"
    ANSWER_GRID = "ANSWER_GRID"
    QR = "QR"
    BARCODE = "BARCODE"


@dataclass
class AnchorPoint:
    x: int
    y: int
    name: str = ""


@dataclass
class BubbleGrid:
    rows: int
    cols: int
    question_start: int
    options: list[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])


@dataclass
class Zone:
    id: str
    name: str
    zone_type: ZoneType
    x: int
    y: int
    width: int
    height: int
    grid: BubbleGrid | None = None

    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


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
        if not (4 <= len(self.anchors) <= 12):
            errors.append("Template must contain 4-12 anchor points.")
        if not self.zones:
            errors.append("Template must contain at least one zone.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for zone in payload["zones"]:
            zone_type = zone.get("zone_type")
            zone["zone_type"] = zone_type.value if hasattr(zone_type, "value") else str(zone_type)
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
                    x=raw["x"],
                    y=raw["y"],
                    width=raw["width"],
                    height=raw["height"],
                    grid=grid,
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
