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

        question_start = int(params.get("question_start", 1))
        total_questions = int(params.get("total_questions", 1))

        if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
            rows = int(params.get("rows", 10))
            cols = int(params.get("columns", 8))
            digit_map = params.get("digit_map", list(range(10)))
            options = [str(d) for d in digit_map[:rows]]
            logical_questions = cols
            semantic = "vertical_digits"

        elif zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
            rows = int(params.get("rows", 10))
            cols = int(params.get("columns", 4))
            digit_map = params.get("digit_map", list(range(10)))
            options = [str(d) for d in digit_map[:rows]]
            logical_questions = cols
            semantic = "vertical_digits"

        elif zone.zone_type == ZoneType.MCQ_BLOCK:
            rows = int(params.get("questions_per_block", 10))
            cols = int(params.get("choices_per_question", 4))
            options = list(params.get("choice_labels", ["A", "B", "C", "D"]))[:cols]
            if len(options) < cols:
                options += [chr(65 + i) for i in range(len(options), cols)]
            logical_questions = rows
            semantic = "row_questions"

        elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            qpb = int(params.get("questions_per_block", 2))
            spq = int(params.get("statements_per_question", 4))
            cps = int(params.get("choices_per_statement", 2))
            rows, cols = qpb * spq, cps
            options = ["Đ", "S"][:cps] if cps <= 2 else [str(i) for i in range(cps)]
            logical_questions = rows
            semantic = "row_statements"

        elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
            rows = int(params.get("rows", 10))
            qpb = int(params.get("questions_per_block", params.get("total_questions", 1)))
            digits = int(params.get("digits_per_answer", 3))
            cols = max(1, qpb * digits)
            sign_row = max(0, int(params.get("sign_row", 1)) - 1)
            decimal_row = max(0, int(params.get("decimal_row", 2)) - 1)
            digit_start_row = max(0, int(params.get("digit_start_row", 3)) - 1)
            digit_map = params.get("digit_map", list(range(max(0, rows - digit_start_row))))
            sign_symbol = str(params.get("sign_symbol", "-"))
            decimal_symbol = str(params.get("decimal_symbol", "."))
            options = ["" for _ in range(rows)]
            if 0 <= sign_row < rows:
                options[sign_row] = sign_symbol
            if 0 <= decimal_row < rows:
                options[decimal_row] = decimal_symbol
            for idx in range(digit_start_row, rows):
                map_idx = idx - digit_start_row
                options[idx] = str(digit_map[map_idx] if map_idx < len(digit_map) else map_idx)
            logical_questions = qpb
            semantic = "numeric_signed_decimal"

        else:
            return None

        scale = float(params.get("grid_scale", 1.0))
        off_x = float(params.get("bubble_offset_x", 0.0))
        off_y = float(params.get("bubble_offset_y", 0.0))

        bubbles: list[tuple[float, float]] = []

        for r in range(rows):
            for c in range(cols):
                u = (c + 0.5) / max(1, cols)
                v = (r + 0.5) / max(1, rows)
                u = min(0.999, max(0.001, (u - 0.5) * scale + 0.5 + off_x))
                v = min(0.999, max(0.001, (v - 0.5) * scale + 0.5 + off_y))
                bubbles.append(self.bilinear_point(control_points, u, v))

        zone.metadata["semantic_layout"] = semantic

        return BubbleGrid(
            rows=rows,
            cols=cols,
            question_start=question_start,
            question_count=logical_questions,
            options=options,
            bubble_positions=bubbles,
        )

    def validate_template(self, template: Template) -> list[str]:
        errors = template.validate()
        profile = str((template.metadata or {}).get("alignment_profile", "auto") or "auto").strip().lower()
        max_anchors = 60 if profile == "one_side" else 20
        if not (4 <= len(template.anchors) <= max_anchors):
            if profile == "one_side":
                errors.append("Template one-side ruler should have 4-60 anchors (top-down row markers on one edge).")
            else:
                errors.append("Template should have 4-20 anchors for robust homography.")
        return errors
