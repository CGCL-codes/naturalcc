import shutil
from collections import Counter

import pytest

from code_agent import security_analysis
from code_agent.scripts.prepare_security_acceptance import generate_cases, statement_count
from code_agent.security_contracts import load_ground_truth, classify_findings


@pytest.mark.parametrize("scan_type,count", [("frequent_defects", 20), ("high_risk", 10)])
def test_fixture_counts_hashes_and_syntax(tmp_path, scan_type, count):
    from code_agent.security_c_rules import parse_source
    truth = generate_cases(tmp_path, scan_type)
    totals = Counter((c["category"], c["expected"]) for c in truth["cases"])
    assert set(totals.values()) == {count}
    for file in truth["source_sha256"]:
        path = tmp_path / file
        assert not parse_source(path.read_text(), path.suffix).has_error
    assert load_ground_truth(tmp_path, "ground_truth.json", scan_type,
        [tmp_path / f for f in truth["source_sha256"]]) == truth
    assert generate_cases(tmp_path, scan_type) == truth


def test_size_counts_statements_instead_of_lines(tmp_path):
    (tmp_path / "a.c").write_text("// comment\n\nint f(){int x=0;\nx++;\nreturn x;\n}\n")
    assert statement_count(tmp_path, ["."])["statement_nodes"] == 2


def test_changed_fixture_is_not_overwritten(tmp_path):
    generate_cases(tmp_path, "high_risk")
    path = tmp_path / "naturalcc_cases/case_0001.c"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_cases(tmp_path, "high_risk")
    assert path.read_text() == "changed"


def test_clang_unavailable_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(security_analysis.shutil, "which", lambda name: None)
    assert security_analysis.clang_scan([str(tmp_path / "x.c")], str(tmp_path), 1)[1]["status"] == "unavailable"


@pytest.mark.skipif(shutil.which("clang") is None, reason="Clang not installed")
def test_real_clang_distinguishes_leak_from_use_after_free_and_string_from_buffer(tmp_path):
    source = tmp_path / "sample.c"
    source.write_text('''#include <stdlib.h>
#include <string.h>
void leak(void) { char *p=malloc(32); if(!p)return; char *q=p; q[0]=1; }
int stale(void) { int *p=malloc(sizeof(int)); if(!p)return 0; free(p); return *p; }
void string(void) { char b[4]="A"; strcat(b,"ABCDEFGH"); }
void bytes(void) { char b[4]; memset(b+1,0,12); }
''')
    findings, coverage = security_analysis.clang_scan([str(source)], str(tmp_path), 1)
    assert coverage["status"] == "completed"
    classify_findings(findings, tmp_path)
    assert {f["category"] for f in findings} >= {"memory_leak", "string_overflow", "buffer_overflow"}
    assert all(f["category"] is None for f in findings if f["diagnostic_type"] == "Use-after-free")
    assert not list(tmp_path.glob("*.plist"))


def test_tsan_runtime_failure_is_not_clean(tmp_path):
    (tmp_path / "tsan.log").write_text("FATAL: ThreadSanitizer: unexpected memory mapping 0x1234\n")
    findings, coverage = security_analysis.import_tsan_report(str(tmp_path), "tsan.log")
    assert findings == []
    assert coverage["status"] == "failed"
    from code_agent.plugins.base import ExecutionContext, PluginResult
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin
    (tmp_path / "main.c").write_text("int main(void){return 0;}")
    context = ExecutionContext(str(tmp_path), ["main.c"], "", "unused", None,
        {"analyzer": "builtin", "scan_type": "high_risk", "sanitizer_report": "tsan.log"})
    result = next(r for r in VulnerabilityDetectionPlugin().execute(context) if isinstance(r, PluginResult))
    assert not result.success
    assert result.artifacts["contract_statistics"]["status"] == "not_evaluated"
    assert "runtime failure" in result.artifacts["contract_statistics"]["reason"]


def test_clang_analyze_keeps_agent_execute_approval(tmp_path):
    from code_agent.agent_core.tool_registry import build_default_registry
    from code_agent.agent_core.contracts import ToolContext
    result = build_default_registry().execute("vulnerability_detection.analyze", {"analyzer": "comprehensive"},
        ToolContext("run", tmp_path, tmp_path / "artifacts"))
    assert result.error["type"] == "ApprovalRequired"
