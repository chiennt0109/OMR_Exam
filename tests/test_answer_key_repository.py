from models.answer_key import AnswerKeyRepository, SubjectKey


def test_get_flexible_matches_exam_code_ignoring_leading_zeroes():
    repo = AnswerKeyRepository()
    repo.upsert(SubjectKey(subject="Hoa_hoc_11", exam_code="0114", answers={1: "A"}))

    assert repo.get_flexible("Hoa_hoc_11", "114") is not None
    assert repo.get_flexible("Hoa_hoc_11", "00114") is not None
    assert repo.get_flexible("Hoa_hoc_11", "0114").exam_code == "0114"
