import json
from pathlib import Path

import pytest

from code_agent.scripts import build_contract5_dataset as build

SAMPLE = """/* header */
#include <rtthread.h>

static int helper(int x);

static int helper(int x)
{
    return x + 1;
}

rt_err_t rt_demo_copy(char *dst, const char *src, rt_size_t size)
{
    if (size > 8)
        return -RT_ERROR;
    rt_memcpy(dst, src, size);
    return RT_EOK;
}
RTM_EXPORT(rt_demo_copy);

int rt_demo_other(void)
{
    return rt_demo_copy(0, 0, 0);
}
"""


def test_function_ranges_use_column_zero_braces_and_skip_prototypes():
    assert build.function_ranges(SAMPLE, "helper") == [(6, 7, 9)]
    assert build.function_ranges(SAMPLE, "rt_demo_copy") == [(11, 12, 17)]
    assert build.function_ranges(SAMPLE, "rt_demo_other") == [(20, 21, 23)]
    assert build.function_ranges(SAMPLE, "rt_memcpy") == []


def _spec_and_upstream(tmp_path: Path, sample: str = SAMPLE) -> tuple[Path, dict]:
    upstream = tmp_path / "rt-thread"
    (upstream / "src").mkdir(parents=True)
    (upstream / "src" / "demo.c").write_text(sample, encoding="utf-8")
    (upstream / "src" / "clean.c").write_text("int clean(void)\n{\n    return 0;\n}\n", encoding="utf-8")
    (upstream / "LICENSE").write_text("Apache\n", encoding="utf-8")
    spec = {"upstream": {"id": "rt-thread", "name": "RT-Thread", "url": "x", "ref": "v", "commit": "c"},
            "project_paths": ["LICENSE", "src"], "clean_files": ["src/clean.c"]}
    return upstream, spec


def _site(**overrides):
    base = {"id": "BOF-01", "file": "src/demo.c", "function": "rt_demo_copy", "expected": "defect", "difficulty": "easy",
            "edits": [{"old": "    if (size > 8)\n        return -RT_ERROR;\n", "new": ""}],
            "marker": "rt_memcpy(dst, src, size);", "description": "d", "trigger": "t", "fix": "f"}
    base.update(overrides)
    return base


def test_resolve_applies_edits_and_reports_lines(tmp_path):
    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"buffer_overflow": {"category": "buffer_overflow", "sites": [
        _site(), _site(id="BOF-C01", function="rt_demo_other", expected="control", edits=[], marker="return rt_demo_copy(0, 0, 0);")]}}
    contents, sites, errors = build.resolve(upstream, spec, injections)
    assert [e for e in errors if "need" not in e] == []  # only the count requirements remain
    assert "if (size > 8)" not in contents["src/demo.c"]
    defect = next(s for s in sites if s["id"] == "BOF-01")
    assert (defect["line_start"], defect["line_end"], defect["line"]) == (11, 15, 13)
    assert defect["modified"] is True and defect["cwe"] == 787
    control = next(s for s in sites if s["id"] == "BOF-C01")
    assert control["modified"] is False and control["expected"] == "control"


@pytest.mark.parametrize("bad, message", [
    ({"edits": [{"old": "return", "new": "return "}]}, "anchor occurs"),
    ({"marker": "return"}, "marker occurs"),
    ({"marker": "return x + 1;"}, "outside"),
    ({"function": "no_such"}, "0 column-0 definitions"),
    ({"expected": "maybe"}, "expected must be"),
    ({"trigger": ""}, "need 'trigger'"),
])
def test_resolve_rejects_bad_sites(tmp_path, bad, message):
    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"buffer_overflow": {"category": "buffer_overflow", "sites": [_site(**bad)]}}
    _, _, errors = build.resolve(upstream, spec, injections)
    assert any(message in e for e in errors), errors


def test_one_site_per_function_and_clean_files_stay_clean(tmp_path):
    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"buffer_overflow": {"category": "buffer_overflow", "sites": [
        _site(), _site(id="BOF-02"),
        _site(id="BOF-03", file="src/clean.c", function="clean", edits=[], marker="return 0;")]}}
    _, _, errors = build.resolve(upstream, spec, injections)
    assert any("one site per function" in e for e in errors)
    assert any("clean file has injected sites" in e for e in errors)


def test_ground_truth_document_marks_file_verdicts(tmp_path):
    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"memory_leak": {"category": "memory_leak", "sites": [
        _site(id="LEAK-01"), _site(id="LEAK-C01", function="rt_demo_other", expected="control", edits=[], marker="return rt_demo_copy(0, 0, 0);")]}}
    contents, sites, _ = build.resolve(upstream, spec, injections)
    files = build.target_files(spec, sites, contents, upstream)
    gt = build.ground_truth_document(spec, sites, files, upstream, contents)
    verdicts = {f["file"]: f["verdict"] for f in gt["target_files"]}
    assert verdicts == {"src/clean.c": "clean", "src/demo.c": "vulnerable"}
    assert gt["categories"]["memory_leak"]["defects"] == 1
    assert gt["project"]["file_count"] == 3
    assert gt["target_file_lines"] == sum(f["lines"] for f in gt["target_files"])
    assert all("line" in s and "line_start" in s for s in gt["sites"])


def test_committed_dataset_is_consistent_when_present():
    if not build.MANIFEST.exists():
        pytest.skip("dataset not built")
    assert build.verify_committed() == []
    gt = json.loads(build.GROUND_TRUTH.read_text(encoding="utf-8"))
    assert gt["target_file_lines"] >= build.MIN_TARGET_LINES
    for category in build.CATEGORIES:
        assert gt["categories"][category]["defects"] == build.SITES_PER_CATEGORY
        assert gt["categories"][category]["controls"] >= build.SITES_PER_CATEGORY
    for site in gt["sites"]:
        text = (build.PROJECT / site["file"]).read_text(encoding="utf-8").splitlines()
        assert site["line_start"] <= site["line"] <= site["line_end"] <= len(text)
        assert site["function"] in text[site["line_start"] - 1]
    functions = [(s["file"], s["function"]) for s in gt["sites"]]
    assert len(functions) == len(set(functions))
    assert not any("injected" in (build.PROJECT / s["file"]).read_text(encoding="utf-8").lower() for s in gt["sites"] if s["modified"])
