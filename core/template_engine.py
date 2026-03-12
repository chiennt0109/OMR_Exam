from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import uuid
import numpy as np

from models.template import BubbleGrid, Template, Zone, ZoneType


@dataclass
class GridSpec:
    rows: int
    cols: int
    h_gap: int = 0
    v_gap: int = 0


class TemplateEngine:
    """Template/grid utilities shared by editor and recognition."""

    def __init__(self, snap_size: int = 5):
        self.snap_size = snap_size

    def snap(self, value: float) -> float:
        return round(value / self.snap_size) * self.snap_size

    def snap_zone(self, zone: Zone) -> Zone:
        zone.x = self.snap(zone.x)
        zone.y = self.snap(zone.y)
        zone.width = max(0.001, self.snap(zone.width))
        zone.height = max(0.001, self.snap(zone.height))
        return zone

    def duplicate_zone(self, zone: Zone, x_offset: float = 0.01, y_offset: float = 0.01) -> Zone:
        copy_zone = deepcopy(zone)
        copy_zone.id = str(uuid.uuid4())
        copy_zone.name = f"{zone.name}_copy"
        copy_zone.x = min(0.99, copy_zone.x + x_offset)
        copy_zone.y = min(0.99, copy_zone.y + y_offset)
        return copy_zone

    def ensure_control_points(self, zone: Zone) -> list[list[float]]:
        corners = zone.metadata.get("control_points")
        if corners and len(corners) == 4:
            return corners
        corners = [
            [zone.x, zone.y],
            [zone.x + zone.width, zone.y],
            [zone.x, zone.y + zone.height],
            [zone.x + zone.width, zone.y + zone.height],
        ]
        zone.metadata["control_points"] = corners
        return corners

    def bilinear_point(self, control_points: list[list[float]], u: float, v: float) -> tuple[float, float]:
        tl, tr, bl, br = [np.array(c, dtype=float) for c in control_points]
        pt = (1 - u) * (1 - v) * tl + u * (1 - v) * tr + (1 - u) * v * bl + u * v * br
        return float(pt[0]), float(pt[1])

    def generate_semantic_grid(self, zone: Zone) -> BubbleGrid | None:
        params = zone.metadata
        control_points = self.ensure_control_points(zone)

        if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
            cols, rows, options, q_count = 6, 10, [str(i) for i in range(10)], 6
        elif zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
            cols, rows, options, q_count = 3, 10, [str(i) for i in range(10)], 3
        elif zone.zone_type == ZoneType.MCQ_BLOCK:
            q_count = int(params.get("total_questions", 10))
            options = list(params.get("choice_labels", ["A", "B", "C", "D"]))
            cols = int(params.get("column_count", 1))
            rows = int(params.get("questions_per_column", max(1, q_count // max(1, cols))))
        elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            qpb = int(params.get("questions_per_block", 2))
            spq = int(params.get("statements_per_question", 4))
            cps = int(params.get("choices_per_statement", 2))
            q_count = qpb * spq
            rows, cols, options = q_count, cps, ["T", "F"][:cps] if cps <= 2 else [str(i) for i in range(cps)]
        elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
            q_count = int(params.get("questions", 5))
            digits = int(params.get("digits_per_answer", 5))
            q_count = q_count * digits
            rows, cols, options = q_count, 10, [str(i) for i in range(10)]
        else:
            return None

        scale = float(params.get("grid_scale", 1.0))
        off_x = float(params.get("bubble_offset_x", 0.0))
        off_y = float(params.get("bubble_offset_y", 0.0))

        bubbles: list[tuple[float, float]] = []
        for q in range(q_count):
            block_col = q // max(1, rows)
            row = q % max(1, rows)
            if block_col >= max(1, cols):
                break
            base_u = (block_col + 0.5) / max(1, cols)
            base_v = (row + 0.5) / max(1, rows)
            for c in range(len(options)):
                du = ((c + 0.5) / max(1, len(options)) - 0.5) * (1 / max(1, cols)) * scale
                u = min(0.999, max(0.001, base_u + du + off_x))
                v = min(0.999, max(0.001, base_v + off_y))
                bubbles.append(self.bilinear_point(control_points, u, v))

        return BubbleGrid(
            rows=max(1, rows),
            cols=max(1, cols),
            question_start=int(params.get("question_start", 1)),
            question_count=max(1, len(bubbles) // max(1, len(options))),
            options=options,
            bubble_positions=bubbles,
        )

    def validate_template(self, template: Template) -> list[str]:
        errors = template.validate()
        if not (4 <= len(template.anchors) <= 12):
            errors.append("Template should have 4-12 anchors for robust homography.")
        return errors
