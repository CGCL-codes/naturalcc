import pytest

from code_agent.scripts import benchmark_contract3_cppcheck as local
from code_agent.scripts import benchmark_contract3_web as web
from code_agent.scripts import render_contract3_report as report


def test_web_answer_must_be_single_category_and_valid_line():
    answer = '{"findings":[{"category":"array_oob","line":3,"reason":"out of bounds"}],"summary":"x"}'
    parsed = web.parse_answer(answer, "array_oob", 5)
    assert parsed["findings"][0]["line"] == 3
    with pytest.raises(ValueError, match="Out-of-scope"):
        web.parse_answer(answer.replace("array_oob", "null_pointer"), "array_oob", 5)
    with pytest.raises(ValueError, match="Invalid source line"):
        web.parse_answer(answer.replace('"line":3', '"line":0'), "array_oob", 5)


def test_web_goal_marks_contract3_as_frequent_defects():
    prompt = web.goal({"id": "array_oob", "name": "数组越界"}, "fixture.c")
    assert 'scan_type:"frequent_defects"' in prompt


def test_web_scoring_is_function_range_bound(tmp_path):
    category = local.load_manifest()["categories"][0]
    _, cases = local.inject_fixture(tmp_path, "fixture", [category])
    positive = next(case for case in cases if case["expected"] == "defect")
    control = next(case for case in cases if case["expected"] == "control")
    prediction = {"findings": [
        {"category": category["id"], "line": positive["defect_line"], "reason": "evidence"},
        {"category": category["id"], "line": control["line_start"], "reason": "incorrect alert"},
        {"category": category["id"], "line": 1, "reason": "outside a function"},
    ], "summary": "x"}
    rows = web.classify(cases, prediction)
    result = web.metric(rows)
    assert result == {"tp": 1, "fn": 19, "fp": 1, "tn": 19, "detection_rate": .05, "false_positive_rate": .05}


def test_console_summary_shows_progress_metrics_and_paths(tmp_path):
    manifest = local.load_manifest()
    rows = [
        {"category": category["id"], "expected": "defect", "detected": True}
        for category in manifest["categories"]
    ]
    checkpoint = {"stop_reason": "测试进行中", "samples": {"sample": {"status": "scored", "rows": rows}}}
    output = web.format_console_summary(checkpoint, manifest, "balanced", tmp_path / "report.md")
    assert "组进度：已评分 1/9" in output
    assert "数组越界" in output
    assert "函数计分：3" in output
    assert "总报告：" in output


def test_completed_records_can_skip_without_a_web_connection():
    manifest = local.load_manifest()
    assert len(web.selected_sample_ids(manifest)) == 9
    assert len(web.selected_sample_ids(manifest, 1)) == 1
    assert web.is_skippable({"status": "scored", "rows": []}, retry_failed=False)
    assert not web.is_skippable({"status": "scored", "rows": []}, retry_failed=False, force_rerun=True)
    assert web.is_skippable({"status": "failed"}, retry_failed=False)
    assert not web.is_skippable({"status": "failed"}, retry_failed=True)
    assert not web.is_skippable({"status": "credit_exhausted"}, retry_failed=False)


def test_only_currently_scored_rows_are_reported():
    checkpoint = {"samples": {
        "completed": {"status": "scored", "rows": [{"expected": "defect", "detected": True}]},
        "failed-rerun": {"status": "failed", "rows": [{"expected": "defect", "detected": True}]},
    }}
    assert web.metric(web.scored_rows(checkpoint))["tp"] == 1
    assert report.web_metrics(checkpoint) == {"n": 1, "tp": 1, "fn": 0, "fp": 0, "tn": 0}
