import json
import shutil
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


def test_ground_truth_document_uses_the_loader_schema(tmp_path):
    """The file must be consumable by security_contracts.load_ground_truth as-is."""
    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"memory_leak": {"category": "memory_leak", "sites": [
        _site(id="LEAK-01"), _site(id="LEAK-C01", function="rt_demo_other", expected="control", edits=[], marker="return rt_demo_copy(0, 0, 0);")]}}
    contents, sites, _ = build.resolve(upstream, spec, injections)
    files = build.target_files(spec, sites, contents, upstream)
    gt = build.ground_truth_document(spec, sites, files, upstream, contents)

    assert gt["schema_version"] == 1 and gt["scan_type"] == "high_risk"
    assert set(gt["source_sha256"]) == {"src/demo.c", "src/clean.c"}
    verdicts = {row["path"]: row["verdict"] for row in gt["detection_files"]}
    assert verdicts == {"src/clean.c": "clean", "src/demo.c": "vulnerable"}
    assert gt["line_totals"]["detection_files"] == sum(r["lines"] for r in gt["detection_files"])
    assert gt["categories"]["memory_leak"]["defects"] == 1

    case_ids = [c["case_id"] for c in gt["cases"]]
    assert case_ids == sorted(set(case_ids)) or len(case_ids) == len(set(case_ids))
    for case in gt["cases"]:
        assert case["category"] in build.CATEGORIES and case["expected"] in build.EXPECTED
        assert case["line_start"] <= case["defect_line"] <= case["line_end"]
        assert case["file"] in gt["source_sha256"]
        assert case["defect_code"] and case["site_id"]


def test_ground_truth_passes_the_product_loader(tmp_path, monkeypatch):
    from code_agent.security_contracts import load_ground_truth

    upstream, spec = _spec_and_upstream(tmp_path)
    injections = {"data_race": {"category": "data_race", "sites": [
        _site(id="RACE-01"), _site(id="RACE-C01", function="rt_demo_other", expected="control", edits=[], marker="return rt_demo_copy(0, 0, 0);")]}}
    contents, sites, _ = build.resolve(upstream, spec, injections)
    files = build.target_files(spec, sites, contents, upstream)
    gt = build.ground_truth_document(spec, sites, files, upstream, contents)

    workspace = tmp_path / "workspace"
    for relative, text in contents.items():
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        (workspace / relative).write_text(text, encoding="utf-8")
    (workspace / "src/clean.c").write_text((upstream / "src/clean.c").read_text(), encoding="utf-8")
    (workspace / "gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    scanned = [str(workspace / row["path"]) for row in gt["detection_files"]]

    loaded = load_ground_truth(str(workspace), "gt.json", "high_risk", scanned)
    assert len(loaded["cases"]) == len(gt["cases"])


def test_committed_dataset_is_consistent_when_present():
    if not build.MANIFEST.exists():
        pytest.skip("dataset not built")
    assert build.verify_committed() == []
    gt = json.loads(build.GROUND_TRUTH.read_text(encoding="utf-8"))
    assert gt["schema_version"] == 1 and gt["scan_type"] == "high_risk"
    assert gt["line_totals"]["detection_files"] >= build.MIN_TARGET_LINES
    for category in build.CATEGORIES:
        assert gt["categories"][category]["defects"] == build.SITES_PER_CATEGORY
        assert gt["categories"][category]["controls"] >= build.SITES_PER_CATEGORY
    for case in gt["cases"]:
        text = (build.PROJECT / case["file"]).read_text(encoding="utf-8").splitlines()
        assert case["line_start"] <= case["defect_line"] <= case["line_end"] <= len(text)
        assert case["function"] in text[case["line_start"] - 1]
    functions = [(c["file"], c["function"]) for c in gt["cases"]]
    assert len(functions) == len(set(functions))
    # The loader rejects overlapping ranges within one (file, category).
    seen: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for case in gt["cases"]:
        key = (case["file"], case["category"])
        assert not any(case["line_start"] <= b and a <= case["line_end"] for a, b in seen.get(key, []))
        seen.setdefault(key, []).append((case["line_start"], case["line_end"]))


def test_compile_database_mirrors_the_sconscript_rules(tmp_path):
    for relative in ("src/ipc.c", "components/drivers/ipc/pipe.c", "components/finsh/shell.c"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("int x;\n")
    spec = {"build_context": {"include_dirs": [".", "include"], "after_dirs": ["libc/include"],
                              "defines": ["__RTTHREAD__"],
                              "local_defines": {"src/": ["__RT_KERNEL_SOURCE__"],
                                                "components/drivers/ipc/": ["__RT_IPC_SOURCE__"]}}}
    entries = {e["file"]: e for e in build.compile_database(spec, tmp_path)}
    assert set(entries) == {"src/ipc.c", "components/drivers/ipc/pipe.c", "components/finsh/shell.c"}
    assert all(e["directory"] == "." for e in entries.values())
    kernel, ipc, shell = (entries[k]["arguments"] for k in ("src/ipc.c", "components/drivers/ipc/pipe.c", "components/finsh/shell.c"))
    assert "-D__RT_KERNEL_SOURCE__" in kernel and "-D__RT_KERNEL_SOURCE__" not in shell
    assert "-D__RT_IPC_SOURCE__" in ipc and "-D__RT_IPC_SOURCE__" not in kernel
    assert shell[:6] == ["cc", "-I.", "-Iinclude", "-idirafter", "libc/include", "-D__RTTHREAD__"]


def test_committed_project_ships_its_compile_context():
    if not build.MANIFEST.exists():
        pytest.skip("dataset not built")
    manifest = json.loads(build.MANIFEST.read_text(encoding="utf-8"))
    shipped = {row["file"]: row for row in manifest["project_files"]}
    assert shipped["rtconfig.h"]["origin"] == "overlay"
    assert shipped["compile_commands.json"]["origin"] == "generated"
    assert (build.PROJECT / "rtconfig.h").read_bytes() == (build.OVERLAY / "rtconfig.h").read_bytes()
    database = json.loads((build.PROJECT / "compile_commands.json").read_text(encoding="utf-8"))
    assert database == build.compile_database(build.load_spec(), build.PROJECT)


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not installed")
def test_every_detection_file_compiles_with_the_shipped_context():
    """Clang and Cppcheck analyse what compiles; a broken context silently loses coverage."""
    if not build.MANIFEST.exists():
        pytest.skip("dataset not built")
    gt = json.loads(build.GROUND_TRUTH.read_text(encoding="utf-8"))
    failures, inactive = build.compile_check(gt)
    assert failures == []
    # Code that needs subsystems outside the dataset (RT-Smart, dynamic modules,
    # the MMU page cache) is compiled out; only these three cases live there.
    assert sorted(line.split()[0] for line in inactive) == ["CE-10", "LEAK-10", "RACE-06"]
