"""Contract-3 exam switches apply only to Contract-3 evaluation, never to other scans."""
from code_agent.plugins.base import ExecutionContext, PluginResult
from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
from code_agent.security_c_rules import scan_c_source
from code_agent import security_exam
from code_agent.security_exam import contract3_exam, switch_off

SOURCE = """#include <string.h>
static int read_it(int *pointer) { return *pointer; }
int via_null_argument(void) { return read_it(0); }
int conditional_assignment(void)
{
    int value = 7;
    int *pointer = 0;
    if (value > 0) pointer = &value;
    return *pointer;
}
"""


def rule_ids(findings):
    return {finding["analyzer_id"] for finding in findings}


def test_full_rules_outside_contract3(monkeypatch):
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"A", "C", "E", "F"}))
    ids = rule_ids(scan_c_source(SOURCE, "a.c", 0)[0])
    assert "nullArgumentDeref" in ids          # E stays on
    assert "nullInitDeref" not in ids          # F's heuristic never runs
    assert not any(switch_off(name) for name in security_exam.SWITCH_NAMES)


def test_exam_rules_inside_contract3_only(monkeypatch):
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"A", "C", "E", "F"}))
    with contract3_exam(True):
        ids = rule_ids(scan_c_source(SOURCE, "a.c", 0)[0])
    assert "nullArgumentDeref" not in ids
    assert "nullInitDeref" in ids
    assert not switch_off("E")                 # scope ended
    with contract3_exam(False):
        assert not switch_off("E")


def plugin_ids(tmp_path, scan_type):
    (tmp_path / "a.c").write_text(SOURCE, encoding="utf-8")
    config = {"analyzer": "builtin", "severity_threshold": "low", "max_findings": 100, "incremental": True}
    if scan_type:
        config["scan_type"] = scan_type
    context = ExecutionContext(str(tmp_path), ["a.c"], "", "unused", None, config)
    result = next(r for r in VulnerabilityDetectionPlugin().execute(context) if isinstance(r, PluginResult))
    return rule_ids(result.artifacts["findings"])


def test_plugin_scopes_switches_to_frequent_defects(tmp_path, monkeypatch):
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"A", "C", "E", "F"}))
    contract3 = plugin_ids(tmp_path, "frequent_defects")
    assert "nullInitDeref" in contract3 and "nullArgumentDeref" not in contract3
    # Same file, same incremental cache: Contract 5 and plain scans keep the full rules.
    for scan_type in ("high_risk", None):
        other = plugin_ids(tmp_path, scan_type)
        assert "nullArgumentDeref" in other and "nullInitDeref" not in other
    assert not switch_off("E")


def test_reset_means_full_rules_for_contract3_too(tmp_path, monkeypatch):
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset())
    ids = plugin_ids(tmp_path, "frequent_defects")
    assert "nullArgumentDeref" in ids and "nullInitDeref" not in ids


def test_reference_full_rules_detect_wrapped_string_overflow_without_flagging_safe_copy(monkeypatch):
    # The full-rules reference is separate from the intentionally incomplete teaching baseline.
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset())
    source = '''#include <string.h>
static void copy_value(char *dst, const char *src) { strcpy(dst, src); }
int unsafe_copy(void) {
    char dst[4] = {0};
    copy_value(dst, "ABCDE");
    return dst[0];
}
int safe_copy(void) {
    char dst[4] = {0};
    copy_value(dst, "ABC");
    return dst[0];
}
'''
    with contract3_exam(True):
        findings = scan_c_source(source, "copies.c", 0)[0]
    overflows = [f for f in findings if f["analyzer_id"] == "capacityStringWrite"]
    assert any(f["line"] == 5 for f in overflows)
    assert not any(8 <= f["line"] <= 12 for f in overflows)


def test_pointer_offset_teaching_gap_is_semantic_and_scoped(tmp_path, monkeypatch):
    from code_agent.security_contracts import classify_findings
    source = """int arbitrary_name(void) {
    int sample[5] = {0};
    int *view = sample + 5;
    return *view;
}
int direct(void) {
    int sample[5] = {0};
    return sample[5];
}
int safe_neighbour(void) {
    int sample[5] = {0};
    int *view = sample + 5;
    return view[-1];
}
"""
    (tmp_path / "unrelated.c").write_text(source)
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"G"}))
    def diagnostics():
        return [{"file": "unrelated.c", "line": n, "analyzer": "cppcheck",
                 "analyzer_id": "arrayIndexOutOfBounds", "rule_id": "cwe-119"} for n in (4, 8, 13)]
    with contract3_exam(True):
        incomplete = classify_findings(diagnostics(), tmp_path)
    # The gap applies equally to a true defect and a hypothetical false positive
    # on a safe pointer access, rather than consulting a vulnerability label.
    assert [f["metric_eligible"] for f in incomplete] == [False, True, False]
    assert len(incomplete) == 3 and incomplete[0]["analyzer_id"] == "arrayIndexOutOfBounds"
    assert "Teaching gap G" in incomplete[0]["classification_basis"]
    for enabled in (False, True):
        if enabled:
            monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset())
        with contract3_exam(enabled):
            assert all(f["metric_eligible"] for f in classify_findings(diagnostics(), tmp_path))


def test_teaching_array_guard_gap_is_scoped_and_name_independent(monkeypatch):
    source = '''int guarded(void) {
    int samples[7] = {0};
    int position = 7;
    if (position >= 7) return 0;
    return samples[position];
}
'''
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"H"}))
    with contract3_exam(True):
        findings = scan_c_source(source, "guard.c", 0)[0]
    assert len(findings) == 1
    assert findings[0]["analyzer_id"] == "guardedArrayAccess"
    assert findings[0]["category"] == "array_oob"
    assert findings[0]["line"] == 5
    assert not scan_c_source(source, "guard.c", 0)[0]
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset())
    with contract3_exam(True):
        assert not scan_c_source(source, "guard.c", 0)[0]


def test_teaching_exact_fit_gap_preserves_true_overflows(monkeypatch):
    source = '''#include <string.h>
int fit(void) { char buffer[7]; strcpy(buffer, "abcdef"); return buffer[0]; }
int spare(void) { char buffer[8]; strcpy(buffer, "abcdef"); return buffer[0]; }
int overflow(void) { char buffer[6]; strcpy(buffer, "abcdef"); return buffer[0]; }
int append_fit(void) { char output[7] = "abc"; strcat(output, "def"); return output[0]; }
'''
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset({"I"}))
    with contract3_exam(True):
        findings = scan_c_source(source, "boundary.c", 0)[0]
    assert {f["line"] for f in findings} == {2, 4, 5}
    assert {f["line"] for f in scan_c_source(source, "boundary.c", 0)[0]} == {4}
    monkeypatch.setattr(security_exam, "CONTRACT3_OFF", frozenset())
    with contract3_exam(True):
        assert {f["line"] for f in scan_c_source(source, "boundary.c", 0)[0]} == {4}
