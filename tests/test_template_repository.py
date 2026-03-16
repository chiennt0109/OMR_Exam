import tempfile
from pathlib import Path

from models.template_repository import TemplateRepository


def test_template_repository_register_and_persist():
    repo = TemplateRepository()
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td) / "math.json"
        tpath.write_text('{"name":"MathFormA","image_path":"x","width":1,"height":1,"anchors":[],"zones":[]}', encoding="utf-8")
        key = repo.register(str(tpath))
        assert key == "MathFormA"
        assert repo.get_path("MathFormA") == str(tpath)

        out = Path(td) / "repo.json"
        repo.save_json(out)
        loaded = TemplateRepository.load_json(out)
        assert loaded.get_path("MathFormA") == str(tpath)
