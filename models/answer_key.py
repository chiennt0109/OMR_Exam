from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class SectionRule:
    name: str
    start_question: int
    end_question: int
    marks_per_correct: float

    def includes(self, question_no: int) -> bool:
        return self.start_question <= question_no <= self.end_question


@dataclass
class SubjectKey:
    subject: str
    exam_code: str
    answers: dict[int, str]
    section_rules: list[SectionRule] = field(default_factory=list)

    def points_for_question(self, question_no: int) -> float:
        for rule in self.section_rules:
            if rule.includes(question_no):
                return rule.marks_per_correct
        return 1.0


@dataclass
class AnswerKeyRepository:
    keys: dict[str, SubjectKey] = field(default_factory=dict)

    @staticmethod
    def _key(subject: str, exam_code: str) -> str:
        return f"{subject}::{exam_code}"

    def upsert(self, key: SubjectKey) -> None:
        self.keys[self._key(key.subject, key.exam_code)] = key

    def get(self, subject: str, exam_code: str) -> SubjectKey | None:
        return self.keys.get(self._key(subject, exam_code))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"keys": {}}
        for k, v in self.keys.items():
            payload["keys"][k] = {
                "subject": v.subject,
                "exam_code": v.exam_code,
                "answers": {str(idx): ans for idx, ans in v.answers.items()},
                "section_rules": [
                    {
                        "name": rule.name,
                        "start_question": rule.start_question,
                        "end_question": rule.end_question,
                        "marks_per_correct": rule.marks_per_correct,
                    }
                    for rule in v.section_rules
                ],
            }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerKeyRepository":
        repo = cls()
        for raw in data.get("keys", {}).values():
            section_rules = [SectionRule(**rule) for rule in raw.get("section_rules", [])]
            repo.upsert(
                SubjectKey(
                    subject=raw["subject"],
                    exam_code=raw["exam_code"],
                    answers={int(k): v for k, v in raw.get("answers", {}).items()},
                    section_rules=section_rules,
                )
            )
        return repo

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "AnswerKeyRepository":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
