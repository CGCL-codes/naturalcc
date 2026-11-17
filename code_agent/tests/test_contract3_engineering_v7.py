"""Diversity, real operation provenance and safe/unsafe mechanism separation."""
import json
from collections import Counter
from pathlib import Path

import pytest

from code_agent.scripts import build_contract3_engineering_v7 as build
from code_agent.security_contracts import classify_findings, load_ground_truth


@pytest.mark.parametrize("project", ["freertos_kernel", "zephyr", "rt_thread"])
@pytest.mark.parametrize("suite", build.SUITES)
def test_diverse_native_snapshot_counts_and_sources(project, suite, tmp_path):
    destination = build.DATA / suite / project
    truth = build.verify(destination)
    counts = Counter((c["category"], c["expected"], c["mechanism"]) for c in truth["cases"])
    assert set(counts.values()) == {2}
    assert len(counts) == 60
    assert not any(c["source_parse_error"] for c in truth["cases"])
    assert len({(c["file"], c["function"]) for c in truth["cases"]
                if c["expected"] == "defect"}) == 60
    assert all(not c["function"].startswith("contract3_") for c in truth["cases"])
    assert all(c["function_start"] <= c["line_start"] <= c["line_end"] <= c["function_end"]
               for c in truth["cases"])
    assert sum(r["lines"] for r in truth["detection_files"]) >= 10000
    # Exercise the product loader with actual source bytes and external GT.
    for file in truth["source_sha256"]:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((destination / "project" / file).read_bytes())
    (tmp_path / "ground_truth.json").write_text(json.dumps(truth))
    loaded = load_ground_truth(tmp_path, "ground_truth.json", "frequent_defects",
                               [tmp_path / f for f in truth["source_sha256"]])
    assert len(loaded["cases"]) == 120


@pytest.mark.parametrize("engine,identifier", [
    ("cppcheck", "arrayIndexOutOfBounds"), ("clang", "alpha.security.ArrayBoundV2")])
def test_external_character_writes_keep_string_category(tmp_path, engine, identifier):
    source = """void work(void) {
    char text[4] = {0};
    char *tail = text;
    int numbers[4] = {0};
    text[4] = '\\0';
    tail[-1] = 'x';
    numbers[4] = 1;
    int value = text[4];
}
"""
    (tmp_path / "renamed.c").write_text(source)
    findings = [dict(file="renamed.c", line=line, analyzer=engine, analyzer_id=identifier)
                for line in (5, 6, 7, 8)]
    classify_findings(findings, tmp_path)
    assert [f["category"] for f in findings] == [
        "string_overflow", "string_overflow", "array_oob", "array_oob"]
    assert all(f["metric_eligible"] for f in findings)


def test_character_write_taxonomy_respects_numeric_shadowing(tmp_path):
    source = "void f(void) { char data[4];\n{ int data[4];\ndata[4]=1; }\ndata[4]='x';\n}\n"
    (tmp_path / "scope.c").write_text(source)
    findings = [dict(file="scope.c", line=line, analyzer="cppcheck", analyzer_id="arrayIndexOutOfBounds")
                for line in (3, 4)]
    classify_findings(findings, tmp_path)
    assert [f['category'] for f in findings] == ['array_oob', 'string_overflow']


def test_pointer_comparison_does_not_mislabel_legal_one_past_address():
    site=dict(destination='entries', capacity='8', address_comparison=True,
              names=dict(offset='position', limit='size', cursor='alias'))
    setup, expression = build.array_mutation(site, 'extent_index', True)
    assert expression == 'entries[8 + 1]'
    setup, expression = build.array_mutation(site, 'extent_index', False)
    assert expression == 'entries[(8) - 1]'
