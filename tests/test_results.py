import json

from gaitlab.config import PipelineConfig
from gaitlab.report.readme_table import END, START, update_readme
from gaitlab.report.results import load_reportable, write_result
from scripts.check_readme_numbers import check


def test_write_result_provenance_and_synthetic_routing(tmp_path):
    p = write_result(
        "grader", "ladder", {"best": {"qwk": 0.5}},
        config=PipelineConfig(), data_manifest="sha256:x", synthetic=True, root=tmp_path,
    )
    assert p.parent == tmp_path / "synthetic" / "grader"
    d = json.loads(p.read_text())
    keys = ("git_sha", "git_dirty", "config_hash", "data_manifest_hash", "hardware", "versions")
    for key in keys:
        assert key in d
    assert d["synthetic"] is True


def test_reportable_excludes_synthetic_and_dirty(tmp_path):
    (tmp_path / "grader").mkdir()
    clean = {"area": "grader", "name": "ladder", "synthetic": False, "git_dirty": False,
             "git_sha": "abc1234", "metrics": {"best": {"qwk": 0.4}}}
    dirty = dict(clean, name="intra_rater", git_dirty=True)
    (tmp_path / "grader" / "ladder.json").write_text(json.dumps(clean))
    (tmp_path / "grader" / "intra_rater.json").write_text(json.dumps(dirty))
    write_result("grader", "verifier", {"pass_rate": 1.0}, config={}, data_manifest="",
                 synthetic=True, root=tmp_path)
    names = {r["name"] for r in load_reportable(tmp_path)}
    assert names == {"ladder"}


def test_readme_block_rewritten_only_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
    (tmp_path / "results" / "grader").mkdir(parents=True)
    (tmp_path / "results" / "grader" / "ladder.json").write_text(json.dumps(
        {"area": "grader", "name": "ladder", "synthetic": False, "git_dirty": False,
         "git_sha": "abc1234", "metrics": {"best": {"qwk": 0.42}}}))
    assert update_readme(readme, tmp_path / "results")
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "0.42" in text and "old" not in text


def test_claim_checker(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text(f"Our grader hits QWK 0.81 on test.\n{START}\nQWK 0.81\n{END}\n")
    assert len(check(str(bad))) == 1
    good = tmp_path / "good.md"
    good.write_text(f"Uses 20-stride windows and Gemma 3 1B.\n{START}\nQWK 0.81, 3.1x\n{END}\n")
    assert check(str(good)) == []


def test_repo_readme_is_clean():
    assert check("README.md") == []
