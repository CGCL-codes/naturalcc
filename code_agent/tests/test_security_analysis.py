import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from code_agent import security_analysis
from code_agent.agent_web_api import app
from code_agent.plugins.base import ExecutionContext, PluginResult
from code_agent.plugins.dispatcher import ExecutionDispatcher
from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin


def run_plugin(tmp_path, **config):
    context = ExecutionContext(str(tmp_path), ["main.c"], "repair", "unused", None, config)
    return [r for r in VulnerabilityDetectionPlugin().execute(context) if isinstance(r, PluginResult)][-1]


def test_cppcheck_maps_bounds_null_and_leak_diagnostics(tmp_path, monkeypatch):
    source = tmp_path / "main.c"
    source.write_text("int f() { return 0; }", encoding="utf-8")
    monkeypatch.setattr(security_analysis.shutil, "which", lambda name: "cppcheck")
    def run(argv, **kwargs):
        assert argv[-1] == str(source.resolve())
        assert kwargs["timeout"] == 120
        assert "shell" not in kwargs
        xml = '<results><errors>' + ''.join(
            f'<error id="{name}" cwe="{cwe}" severity="error" msg="{name}"><location file="main.c" line="1"/></error>'
            for name, cwe in [("arrayIndexOutOfBounds", "788"), ("nullPointer", "476"), ("memleak", "401")]
        ) + '</errors></results>'
        return SimpleNamespace(returncode=0, stderr=xml)
    monkeypatch.setattr(security_analysis.subprocess, "run", run)
    findings, coverage = security_analysis.cppcheck_scan([str(source)], str(tmp_path), 1)
    assert {f["rule_id"] for f in findings} == {"cwe-788", "cwe-476", "cwe-401"}
    assert coverage["status"] == "completed"


@pytest.mark.parametrize("stderr,code,status", [("not XML", 0, "failed"), ("failed", 1, "failed"), ('<results><errors><error id="syntaxError" msg="bad syntax"/></errors></results>', 0, "partial")])
def test_analyzer_errors_are_not_clean_reports(tmp_path, monkeypatch, stderr, code, status):
    source = tmp_path / "main.c"
    source.write_text("invalid", encoding="utf-8")
    monkeypatch.setattr(security_analysis.shutil, "which", lambda name: "cppcheck")
    monkeypatch.setattr(security_analysis.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stderr=stderr))
    _, coverage = security_analysis.cppcheck_scan([str(source)], str(tmp_path), 1)
    assert coverage["status"] == status


def test_required_analyzer_missing_is_failure(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("int a;", encoding="utf-8")
    monkeypatch.setattr(security_analysis.shutil, "which", lambda name: None)
    result = run_plugin(tmp_path, analyzer="cppcheck")
    assert not result.success
    assert result.artifacts["coverage"][1]["status"] == "unavailable"


def test_incremental_reuses_findings_and_invalidates_edits_deletions_and_settings(tmp_path):
    first, second = tmp_path / "main.c", tmp_path / "other.c"
    first.write_text("strcpy(buf, input);\n", encoding="utf-8")
    second.write_text("printf(input);\n", encoding="utf-8")
    def scan(files, threshold="medium", max_findings=30):
        plugin = VulnerabilityDetectionPlugin()
        result = plugin._scan_files([str(p) for p in files], threshold, "default", max_findings, str(tmp_path), 1, incremental=True)
        return result, plugin.last_scan_stats
    original, stats = scan([first, second], max_findings=1)
    assert stats == {"scanned_files": 2, "reused_files": 0}
    expanded, stats = scan([first, second])
    assert len(expanded) == 2
    assert stats["reused_files"] == 2
    first.write_text("int safe;\n", encoding="utf-8")
    changed, stats = scan([first, second])
    assert stats == {"scanned_files": 1, "reused_files": 1}
    assert all(f["file"] == "other.c" for f in changed)
    second.unlink()
    assert scan([first])[0] == []
    assert scan([first], "critical")[1]["scanned_files"] == 1


def test_result_limit_reports_all_filtered_candidates_without_changing_order(tmp_path):
    (tmp_path / "main.c").write_text("strcpy(a, b);\nprintf(input);\nstrcat(a, b);\n", encoding="utf-8")
    result = run_plugin(tmp_path, analyzer="builtin", max_findings=1)
    summary = result.artifacts["finding_summary"]
    assert summary == {"candidate_count": 3, "returned_count": 1, "truncated": True, "max_findings": 1}
    assert result.artifacts["findings"][0]["rule_id"] == "cwe-120"
    assert "Total findings:" not in result.report
    assert "Candidates after severity threshold: 3" in result.report
    assert "Returned findings: 1" in result.report
    assert "Findings truncated: yes" in result.report


def test_scan_type_preserves_findings_and_requires_ground_truth_for_rates(tmp_path):
    (tmp_path / "main.c").write_text("strcpy(a, b);\n", encoding="utf-8")
    plugin = VulnerabilityDetectionPlugin()
    assert plugin.validate({"scan_type": ""}) is None
    assert plugin.validate({"scan_type": "frequent_defects"}) is None
    assert plugin.validate({"scan_type": "high_risk"}) is None
    assert "scan_type" in plugin.validate({"scan_type": "unknown"})
    assert "scan_type" in plugin.validate({"scan_type": []})
    baseline = run_plugin(tmp_path, analyzer="builtin")
    selected = run_plugin(tmp_path, analyzer="builtin", scan_type="high_risk")
    assert selected.artifacts["findings"] == baseline.artifacts["findings"]
    assert selected.artifacts["scan_type"] == "high_risk"
    assert selected.artifacts["contract_statistics"]["status"] == "not_evaluated"
    assert "ground truth" in selected.artifacts["contract_statistics"]["reason"]
    assert "finding-matching rules" in selected.artifacts["contract_statistics"]["reason"]
    assert "Scan type: high_risk" in selected.report
    assert "Contract statistics: not_evaluated" in selected.report
    assert "ground truth" in selected.report
    assert "scan_type" not in baseline.artifacts
    assert "Scan type:" not in baseline.report
    assert "scan_type" not in run_plugin(tmp_path, analyzer="builtin", scan_type="").artifacts


def test_result_limit_counts_builtin_cppcheck_and_tsan_after_threshold(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("strcpy(a, b);\n", encoding="utf-8")
    (tmp_path / "tsan.log").write_text("WARNING: ThreadSanitizer: data race\n  #0 worker main.c:1:3\n", encoding="utf-8")
    cpp_finding = {"severity": "high", "confidence": "high", "rule_id": "cwe-476", "rule_name": "null", "file": "main.c", "line": 1, "snippet": "*p", "recommendation": "fix"}
    monkeypatch.setattr(security_analysis, "cppcheck_scan", lambda *args: ([cpp_finding], {"engine": "cppcheck", "status": "completed"}))
    result = run_plugin(tmp_path, analyzer="auto", sanitizer_report="tsan.log", max_findings=1, severity_threshold="high")
    assert result.artifacts["finding_summary"] == {"candidate_count": 3, "returned_count": 1, "truncated": True, "max_findings": 1}
    assert len(result.artifacts["findings"]) == 1
    assert [row["engine"] for row in result.artifacts["coverage"]] == ["builtin", "cppcheck", "tsan-import"]


def test_incremental_cache_does_not_understate_more_than_1000_candidates(tmp_path):
    (tmp_path / "main.c").write_text("strcpy(a, b);\n" * 1001, encoding="utf-8")
    first = run_plugin(tmp_path, analyzer="builtin", incremental=True, max_findings=1)
    second = run_plugin(tmp_path, analyzer="builtin", incremental=True, max_findings=1)
    assert first.artifacts["finding_summary"]["candidate_count"] == 1001
    assert second.artifacts["finding_summary"]["candidate_count"] == 1001
    assert second.artifacts["coverage"][0]["scanned_files"] == 0
    assert second.artifacts["coverage"][0]["reused_files"] == 1
    expanded = run_plugin(tmp_path, analyzer="builtin", incremental=True, max_findings=1000)
    assert expanded.artifacts["finding_summary"] == {
        "candidate_count": 1001, "returned_count": 1000, "truncated": True, "max_findings": 1000}
    assert [finding["line"] for finding in expanded.artifacts["findings"]] == list(range(1, 1001))


def test_pipeline_stream_accepts_scan_type_and_reports_invalid_value(tmp_path):
    (tmp_path / "main.c").write_text("strcpy(a, b);\n", encoding="utf-8")
    def events(scan_type):
        context = ExecutionContext(str(tmp_path), ["main.c"], "", "unused", None,
            {"feature": "vulnerability_detection", "analyzer": "builtin", "scan_type": scan_type})
        return [json.loads(event) for event in ExecutionDispatcher().dispatch(context)]
    accepted = events("frequent_defects")
    assert accepted[-1]["type"] == "done"
    assert accepted[-1]["status"] == "success"
    assert "Contract statistics: not_evaluated" in accepted[-1]["report"]
    invalid = events("unknown")
    assert invalid == [{"type": "error", "status": "error",
        "log": "Validation error: scan_type must be frequent_defects or high_risk"}]


def test_legacy_run_accepts_scan_type_in_feature_config(tmp_path):
    (tmp_path / "main.c").write_text("strcpy(a, b);\n", encoding="utf-8")
    payload = {"project_dir": str(tmp_path), "target_files": ["main.c"],
        "feature": "vulnerability_detection", "feature_config": {"analyzer": "builtin", "scan_type": "high_risk"}}
    response = TestClient(app).post("/api/run", json=payload)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["status"] == "success"
    assert "Scan type: high_risk" in events[-1]["report"]
    assert "Contract statistics: not_evaluated" in events[-1]["report"]


def test_thread_rule_is_review_hint_and_tsan_import_is_distinguished(tmp_path):
    (tmp_path / "main.c").write_text("pthread_create(&t, 0, worker, 0);\nlocaltime(&now);\n", encoding="utf-8")
    result = run_plugin(tmp_path, analyzer="builtin")
    race = next(f for f in result.artifacts["findings"] if f["rule_id"] == "cwe-362")
    assert race["confidence"] == "low"
    (tmp_path / "tsan.log").write_text("WARNING: ThreadSanitizer: data race\n  #0 worker main.c:2:3\n", encoding="utf-8")
    findings, coverage = security_analysis.import_tsan_report(str(tmp_path), "tsan.log")
    assert findings[0]["file"] == "main.c"
    assert findings[0]["analyzer"] == "tsan-import"
    assert "not verified" in findings[0]["evidence"]
    assert coverage["status"] == "imported"
    (tmp_path / "main.c").write_text("localtime(&now);\n", encoding="utf-8")
    assert not run_plugin(tmp_path, analyzer="builtin").artifacts["findings"]


def test_scanner_rejects_outside_targets_and_reports(tmp_path):
    outside = tmp_path / "outside.c"
    outside.write_text("secret", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError):
        VulnerabilityDetectionPlugin()._collect_files(str(workspace), [str(outside)], "targets")
    with pytest.raises(ValueError):
        security_analysis.import_tsan_report(str(workspace), "../outside.c")


def test_failed_aider_is_not_reported_as_success(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("strcpy(buf, input);\n", encoding="utf-8")
    monkeypatch.setattr("code_agent.plugins.vulnerability_detection.run_aider_stream", lambda **kw: iter(["Aider failed"]))
    result = run_plugin(tmp_path, analyzer="builtin", auto_fix=True)
    assert not result.success


def test_vulnerability_auto_fix_passes_custom_ollama_base(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("strcpy(buf, input);\n", encoding="utf-8")
    called = []
    monkeypatch.setattr("code_agent.plugins.vulnerability_detection.run_aider_stream", lambda **kwargs:
        called.append(kwargs) or iter(["任务圆满完成"]))
    context = ExecutionContext(str(tmp_path), ["main.c"], "repair", "ollama_chat/qwen2.5-coder:1.5b",
        None, {"analyzer": "builtin", "auto_fix": True}, base_url="http://localhost:11501/v1")
    result = [item for item in VulnerabilityDetectionPlugin().execute(context) if isinstance(item, PluginResult)][-1]
    assert result.success
    assert called[0]["base_url"] == "http://localhost:11501/v1"


@pytest.mark.skipif(shutil.which("cppcheck") is None, reason="Cppcheck not installed on this host")
def test_real_cppcheck_bounds_null_and_leak(tmp_path):
    source = tmp_path / "main.c"
    source.write_text("#include <stdlib.h>\nint bounds(void) { int a[2]; a[3]=1; return a[3]; }\nint null(void) { int *p=0; return *p; }\nvoid leak(void) { int *p=malloc(20); if (!p) return; p[0]=1; }\n", encoding="utf-8")
    findings, coverage = security_analysis.cppcheck_scan([str(source)], str(tmp_path), 1)
    ids = {f["analyzer_id"] for f in findings}
    assert {"arrayIndexOutOfBounds", "nullPointer", "memleak"} <= ids
    assert coverage["status"] in {"completed", "partial"}
