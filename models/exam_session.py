from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from models.answer_key import AnswerKeyRepository
from models.database import OMRDatabase


@dataclass
class Student:
    student_id: str
    name: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExamSession:
    exam_name: str
    exam_date: str
    subjects: list[str]
    template_path: str
    answer_key_path: str
    students: list[Student] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def add_student(self, student: Student) -> None:
        self.students.append(student)

    def import_students_csv(self, path: str | Path) -> int:
        import csv

        count = 0
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = str(row.get("StudentID", row.get("student_id", ""))).strip()
                name = str(row.get("Name", row.get("name", ""))).strip()
                if not sid:
                    continue
                self.students.append(Student(student_id=sid, name=name, extra=row))
                count += 1
        return count

    def import_students_excel(self, path: str | Path) -> int:
        import pandas as pd

        df = pd.read_excel(path)
        count = 0
        for _, row in df.iterrows():
            sid = str(row.get("StudentID", row.get("student_id", ""))).strip()
            name = str(row.get("Name", row.get("name", ""))).strip()
            if not sid:
                continue
            self.students.append(Student(student_id=sid, name=name, extra=row.to_dict()))
            count += 1
        return count

    def load_answer_keys(self) -> AnswerKeyRepository:
        return AnswerKeyRepository.load_json(self.answer_key_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exam_name": self.exam_name,
            "exam_date": self.exam_date,
            "subjects": self.subjects,
            "template_path": self.template_path,
            "answer_key_path": self.answer_key_path,
            "students": [
                {"student_id": s.student_id, "name": s.name, "extra": s.extra}
                for s in self.students
            ],
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExamSession":
        session = cls(
            exam_name=data["exam_name"],
            exam_date=data.get("exam_date", str(date.today())),
            subjects=list(data.get("subjects", [])),
            template_path=data["template_path"],
            answer_key_path=data["answer_key_path"],
        )
        session.students = [Student(**s) for s in data.get("students", [])]
        session.config = dict(data.get("config", {}))
        return session

    def save_json(self, path: str | Path) -> None:
        path_obj = Path(path)
        session_id = path_obj.stem or path_obj.name or self.exam_name
        OMRDatabase.default().save_exam_session(session_id, self.exam_name, self.to_dict())

    @classmethod
    def load_json(cls, path: str | Path) -> "ExamSession":
        path_obj = Path(path)
        session_id = path_obj.stem or path_obj.name
        payload = OMRDatabase.default().fetch_exam_session(session_id)
        if not payload:
            raise FileNotFoundError(f"Session '{session_id}' not found in SQLite storage.")
        return cls.from_dict(payload)
