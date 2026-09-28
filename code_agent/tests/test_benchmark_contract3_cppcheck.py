from pathlib import Path

from code_agent.scripts import benchmark_contract3_cppcheck as bench


def test_manifest_declares_three_pinned_sources_and_categories():
    manifest = bench.load_manifest()
    assert len(manifest["sources"]) == len(manifest["categories"]) == 3
    assert all(len(source["commit"]) == 40 and source["scan_roots"] for source in manifest["sources"])


def test_fixture_has_twenty_defects_and_controls_per_category(tmp_path):
    manifest = bench.load_manifest()
    fixture, cases = bench.inject_fixture(tmp_path, "fixture", manifest["categories"])
    assert fixture.exists()
    assert len(cases) == 120
    content = fixture.read_text(encoding="utf-8")
    assert "_bad_" not in content
    assert "_control_" not in content
    for category in manifest["categories"]:
        rows = [case for case in cases if case["category"] == category["id"]]
        assert sum(row["expected"] == "defect" for row in rows) == 20
        assert sum(row["expected"] == "control" for row in rows) == 20


def test_metrics_and_category_matching_are_location_bound():
    manifest = bench.load_manifest()
    categories = manifest["categories"]
    cases = [
        {"category": "array_oob", "expected": "defect", "file": "x.c", "line_start": 1, "line_end": 3},
        {"category": "array_oob", "expected": "defect", "file": "x.c", "line_start": 4, "line_end": 6},
        {"category": "array_oob", "expected": "control", "file": "x.c", "line_start": 7, "line_end": 9},
        {"category": "array_oob", "expected": "control", "file": "x.c", "line_start": 10, "line_end": 12},
    ]
    findings = [
        {"file": "x.c", "line": 2, "analyzer_id": "arrayIndexOutOfBounds", "rule_id": "cwe-788"},
        {"file": "x.c", "line": 8, "analyzer_id": "arrayIndexOutOfBounds", "rule_id": "cwe-788"},
        {"file": "x.c", "line": 20, "analyzer_id": "arrayIndexOutOfBounds", "rule_id": "cwe-788"},
    ]
    rows = bench.classify(cases, findings, categories)
    assert bench.metric(rows) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1, "detection_rate": .5, "false_positive_rate": .5}
