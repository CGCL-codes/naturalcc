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
