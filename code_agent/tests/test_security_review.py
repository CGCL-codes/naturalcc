import pytest

from code_agent.security_review import REVIEW_GUIDANCE, REVIEW_VERSION, unbounded_scanf_lines
from code_agent.agent_core.contracts import ToolContext
from code_agent.agent_core.tool_registry import build_default_registry


@pytest.mark.parametrize("source", [
    'scanf("%s", user);',
    'scanf("%ls", wide);',
    'scanf("%[^;]", user);',
    'fscanf(stream, "%s", user);',
    'sscanf(input, "%s", user);',
    'scanf("%*s %s", user);',
    'scanf("%% %s", user);',
    'scanf("name=" "%s", user);',
    'fscanf(get_stream(a, b), "%s", user);',
])
def test_unbounded_string_formats_are_candidates(source):
    assert unbounded_scanf_lines(source) == {1}


@pytest.mark.parametrize("source", [
    'scanf("%23s", user);',
    'scanf("%23ls", wide);',
    'scanf("%23[^;]", user);',
    'scanf("%*s", user);',
    'scanf("%%s", user);',
    'scanf("%c %d", ch, value);',
    'sscanf("%s", "%2s", user);',
    'printf("scanf(\\"%s\\", user)");',
    '// scanf("%s", user);',
    '/*\n scanf("%s", user);\n */',
    'scanf(format_from_caller, user);',
])
def test_bounded_suppressed_nonwriting_and_unknown_formats_are_not_candidates(source):
    assert unbounded_scanf_lines(source) == set()


def test_multiline_format_and_original_line_number():
    source = '// comment\n\nscanf(\n /* format */ "%s",\n user\n);\n'
    assert unbounded_scanf_lines(source) == {3}


def test_real_agent_tool_receives_guidance_and_candidate_without_changing_source(tmp_path):
    source = '#include <stdio.h>\nint f(void) { char input[24]; scanf("%s", input); return 0; }\n'
    (tmp_path / "input.c").write_text(source)
    result = build_default_registry(False).execute(
        "vulnerability_detection", {"target_files": ["input.c"], "rule_profile": "c_cpp"},
        ToolContext("review-test", tmp_path, tmp_path / "artifacts"))
    assert result.status == "success"
    assert result.changed_files == []
    assert (tmp_path / "input.c").read_text() == source
    assert [(f["rule_id"], f["line"]) for f in result.data["findings"]] == [("cwe-787", 2)]
    review = result.data["review_guidance"]
    assert review == {"version": REVIEW_VERSION, "checks": REVIEW_GUIDANCE}
    assert "not additional findings" in result.data["report"]
    assert "CWE-479/833" in review["checks"]["concurrency"]
    assert "integer representation" in review["checks"]["commands"]
    assert "n-1" in review["checks"]["bounds"]


def test_bounded_read_needs_review_but_no_unbounded_scanf_alarm(tmp_path):
    (tmp_path / "input.c").write_text('int f(void) { char input[24]; scanf("%23s", input); }')
    result = build_default_registry(False).execute(
        "vulnerability_detection", {"target_files": ["input.c"]},
        ToolContext("bounded-test", tmp_path, tmp_path / "artifacts"))
    assert not result.data["findings"]
    assert result.data["review_guidance"]["version"] == REVIEW_VERSION


def test_non_c_scan_does_not_inject_c_specific_review(tmp_path):
    (tmp_path / "input.py").write_text('print("hello")')
    result = build_default_registry(False).execute(
        "vulnerability_detection", {"target_files": ["input.py"]},
        ToolContext("python-test", tmp_path, tmp_path / "artifacts"))
    assert result.status == "success"
    assert result.data["review_guidance"] == {}
