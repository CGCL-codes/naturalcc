import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_agent import security_analysis
from code_agent.plugins.base import ExecutionContext, PluginResult
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


@pytest.mark.skipif(shutil.which("cppcheck") is None, reason="Cppcheck not installed on this host")
def test_real_cppcheck_bounds_null_and_leak(tmp_path):
    source = tmp_path / "main.c"
    source.write_text("#include <stdlib.h>\nint bounds(void) { int a[2]; a[3]=1; return a[3]; }\nint null(void) { int *p=0; return *p; }\nvoid leak(void) { int *p=malloc(20); if (!p) return; p[0]=1; }\n", encoding="utf-8")
    findings, coverage = security_analysis.cppcheck_scan([str(source)], str(tmp_path), 1)
    ids = {f["analyzer_id"] for f in findings}
    assert {"arrayIndexOutOfBounds", "nullPointer", "memleak"} <= ids
    assert coverage["status"] in {"completed", "partial"}
