from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import uuid

from models.template import BubbleGrid, Template, Zone, ZoneType


@dataclass
class GridSpec:
    rows: int
    cols: int
    h_gap: int = 0
    v_gap: int = 0


class TemplateEngine:
    """Handles template manipulation logic used by the editor UI."""

    def __init__(self, snap_size: int = 5):
        self.snap_size = snap_size

    def snap(self, value: int) -> int:
        return round(value / self.snap_size) * self.snap_size

    def snap_zone(self, zone: Zone) -> Zone:
        zone.x = self.snap(zone.x)
        zone.y = self.snap(zone.y)
        zone.width = self.snap(zone.width)
        zone.height = self.snap(zone.height)
        return zone

    def duplicate_zone(self, zone: Zone, x_offset: int = 10, y_offset: int = 10) -> Zone:
        copy_zone = deepcopy(zone)
        copy_zone.id = str(uuid.uuid4())
        copy_zone.name = f"{zone.name}_copy"
        copy_zone.x += x_offset
        copy_zone.y += y_offset
        return copy_zone

    def generate_auto_grid(
        self,
        base_name: str,
        zone_type: ZoneType,
        x: int,
        y: int,
        cell_width: int,
        cell_height: int,
        spec: GridSpec,
        question_start: int = 1,
    ) -> list[Zone]:
        zones: list[Zone] = []
        q = question_start
        for r in range(spec.rows):
            for c in range(spec.cols):
                zx = x + c * (cell_width + spec.h_gap)
                zy = y + r * (cell_height + spec.v_gap)
                grid = BubbleGrid(rows=1, cols=5, question_start=q)
                zones.append(
                    Zone(
                        id=str(uuid.uuid4()),
                        name=f"{base_name}_{q}",
                        zone_type=zone_type,
                        x=zx,
                        y=zy,
                        width=cell_width,
                        height=cell_height,
                        grid=grid,
                    )
                )
                q += 1
        return zones

    def validate_template(self, template: Template) -> list[str]:
        return template.validate()
