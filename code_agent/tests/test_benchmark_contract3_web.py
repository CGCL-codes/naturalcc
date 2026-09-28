import pytest

from code_agent.scripts import benchmark_contract3_cppcheck as local
from code_agent.scripts import benchmark_contract3_web as web


def test_web_answer_must_be_single_category_and_valid_line():
    answer = '{"findings":[{"category":"array_oob","line":3,"reason":"out of bounds"}],"summary":"x"}'
    parsed = web.parse_answer(answer, "array_oob", 5)
    assert parsed["findings"][0]["line"] == 3
    with pytest.raises(ValueError, match="Out-of-scope"):
        web.parse_answer(answer.replace("array_oob", "null_pointer"), "array_oob", 5)
    with pytest.raises(ValueError, match="Invalid source line"):
        web.parse_answer(answer.replace('"line":3', '"line":0'), "array_oob", 5)


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
