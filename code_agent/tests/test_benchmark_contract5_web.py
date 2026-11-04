import json

import pytest

from code_agent.scripts import benchmark_contract5_web as bench

CASES = [
    {"case_id": "rt-thread-buffer_overflow-defect-01", "site_id": "BOF-01", "project": "rt-thread",
     "category": "buffer_overflow", "cwe": 787, "file": "src/a.c", "function": "f_bad", "line_start": 10,
     "line_end": 20, "defect_line": 15, "defect_code": "x", "expected": "defect", "difficulty": "easy",
     "description": "d", "trigger": "t", "fix": "x"},
    {"case_id": "rt-thread-buffer_overflow-control-01", "site_id": "BOF-C01", "project": "rt-thread",
     "category": "buffer_overflow", "cwe": 787, "file": "src/a.c", "function": "f_ok", "line_start": 30,
     "line_end": 40, "defect_line": 35, "defect_code": "x", "expected": "control", "difficulty": "medium",
     "description": "d", "trigger": "", "fix": ""},
    {"case_id": "rt-thread-data_race-defect-01", "site_id": "RACE-01", "project": "rt-thread",
     "category": "data_race", "cwe": 362, "file": "src/a.c", "function": "g_bad", "line_start": 50,
     "line_end": 60, "defect_line": 55, "defect_code": "x", "expected": "defect", "difficulty": "hard",
     "description": "d", "trigger": "t", "fix": "x"},
]
GT = {"schema_version": 1, "scan_type": "high_risk",
      "project": {"name": "RT-Thread", "ref": "v5.2.1", "commit": "97893c004c65", "file_count": 3, "source_lines": 300},
      "detection_files": [
          {"path": "src/a.c", "lines": 100, "sha256": "", "verdict": "vulnerable", "defects": 2, "controls": 1,
           "categories": ["buffer_overflow", "data_race"]},
          {"path": "src/b.c", "lines": 50, "sha256": "", "verdict": "clean", "defects": 0, "controls": 0,
           "categories": []}],
      "line_totals": {"detection_files": 150, "project_source": 300},
      "source_sha256": {"src/a.c": "", "src/b.c": ""},
      "categories": {c: {"name": n, "cwe": 0, "defects": 0, "controls": 0} for c, n in bench.CATEGORIES.items()},
      "cases": CASES}


def test_prompt_template_has_placeholder_and_no_labels():
    text = bench.goal("src/thread.c")
    assert "`src/thread.c`" in text and "{file}" not in text
    for secret in ("ground_truth", "BOF-", "RACE-", "LEAK-", "CE-", "注入"):
        assert secret not in text


def test_parse_answer_tolerates_stray_fences_and_prose():
    """A correct answer must never be thrown away over formatting noise."""
    body = '{"findings":[],"summary":"safe"}'
    for wrapper in (body,
                    f"json\n{body}",                      # fence stripped, language tag left behind
                    f"```json\n{body}\n```",
                    f"```\n{body}\n```",
                    f"结论如下：\n{body}\n以上。",          # surrounded by prose
                    f"JSON\n{body}"):
        assert bench.parse_answer(wrapper, 10)["findings"] == []
    # Braces inside strings must not end the object early.
    tricky = '{"findings":[],"summary":"a } brace { inside"}'
    assert bench.parse_answer(f"note\n{tricky}", 10)["summary"] == "a } brace { inside"
    # A runtime-context dump whose first object is not an answer stays unparseable.
    with pytest.raises(ValueError):
        bench.parse_answer('[ACTIVE_STATE]\n{"approval_state": null, "changed_files": []}', 10)


def test_parse_answer_accepts_categories_or_cwe_and_rejects_bad_output():
    ok = bench.parse_answer('```json\n{"findings":[{"category":"data_race","function":"g","line":3,"reason":"r"}],"summary":"s"}\n```', 10)
    assert ok["findings"][0]["category"] == "data_race"
    mapped = bench.parse_answer('{"findings":[{"cwe":401,"line":3,"reason":"r"}],"summary":"s"}', 10)
    assert mapped["findings"][0]["category"] == "memory_leak"
    assert bench.parse_answer('text\n```json\n{"findings":[],"summary":"safe"}\n```', 10)["findings"] == []
    for bad in ("no bugs", "{}", '{"findings":[],"summary":null}',
                '{"findings":[{"category":"null_pointer","line":1,"reason":"r"}],"summary":"s"}',
                '{"findings":[{"category":"data_race","line":11,"reason":"r"}],"summary":"s"}',
                '{"findings":[{"category":"data_race","line":1,"reason":""}],"summary":"s"}',
                '```json\n{"findings":[],"summary":"a"}\n```\n```json\n{"findings":[],"summary":"b"}\n```'):
        with pytest.raises((ValueError, TypeError)):
            bench.parse_answer(bad, 10)


def test_classify_matches_by_function_or_line_and_keeps_category_strict():
    prediction = {"findings": [
        {"category": "buffer_overflow", "function": "f_bad", "line": 99, "reason": "by name"},
        {"category": "data_race", "function": "other", "line": 35, "reason": "wrong category on control"},
        {"category": "memory_leak", "function": "elsewhere", "line": 80, "reason": "out of scope"}]}
    rows, unmatched = bench.classify(CASES, prediction)
    by_id = {r["site_id"]: r["detected"] for r in rows}
    assert by_id == {"BOF-01": True, "BOF-C01": False, "RACE-01": False}
    assert [f["reason"] for f in unmatched] == ["wrong category on control", "out of scope"]
    m = bench.metric(rows)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 0, 1)
    assert m["warning_accuracy"] == 1.0 and m["false_positive_rate"] == 0.0 and m["accuracy"] == pytest.approx(2 / 3)
    assert bench.metric([])["accuracy"] is None and bench.rate(None) == "N/A"


def test_file_metric_and_ordering():
    targets = bench.ordered_targets(GT)
    assert [t["file"] for t in targets] == ["src/a.c", "src/b.c"]
    records = {"src/a.c": {"prediction": {"findings": [{"category": "data_race", "line": 1}]}},
               "src/b.c": {"prediction": {"findings": [{"category": "memory_leak", "line": 1}]}}}
    fm = bench.file_metric(records, targets)
    assert (fm["tp"], fm["tn"], fm["fp"], fm["fn"]) == (1, 0, 1, 0)
    assert fm["false_positive_rate"] == 1.0


def test_report_renders_without_results_and_with_results(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUT", tmp_path)
    empty = {"samples": {}, "stop_reason": "尚未开始"}
    text = bench.report(empty, GT, "anthropic/claude-sonnet-4.5")
    assert "尚未产生任何有效评分结果" in text and "N/A" in text
    prediction = {"findings": [{"category": "buffer_overflow", "function": "f_ok", "line": 35, "reason": "looks unsafe"}], "summary": "s"}
    rows, unmatched = bench.classify(CASES, prediction)
    checkpoint = {"samples": {"src/a.c": {"status": "scored", "prediction": prediction, "rows": rows, "out_of_scope_findings": unmatched,
                                           "attempts": [{"run_id": "run-1234"}]}}, "stop_reason": "x"}
    text = bench.report(checkpoint, GT, "anthropic/claude-sonnet-4.5")
    assert "| 预警准确率（合同口径） | TP/(TP+FP) | 0.00% (0/1) |" in text
    assert "| 误报率（合同口径） | FP/(FP+TN) | 100.00% (1/1) |" in text
    assert "`BOF-C01`" in text and "误报 FP" in text and "`BOF-01`" in text and "漏报 FN" in text
    assert "run-1234"[:8] in text


def test_offline_rescore_uses_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUT", tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/run.json").write_text(json.dumps({"state": {"status": "completed", "final_answer":
        '```json\n{"findings":[{"category":"data_race","function":"g_bad","line":55,"reason":"r"}],"summary":"s"}\n```'}}))
    checkpoint = {"samples": {"src/a.c": {"status": "unscored", "error": "x", "attempts": [{"run_id": "run"}]}}}
    bench.refresh_scores(checkpoint, GT)
    rec = checkpoint["samples"]["src/a.c"]
    assert rec["status"] == "scored" and "error" not in rec
    assert {r["site_id"]: r["detected"] for r in rec["rows"]} == {"BOF-01": False, "BOF-C01": False, "RACE-01": True}


def test_credit_failure_only_uses_runtime_errors():
    assert bench.credit_failure([{"type": "run.failed", "payload": {"error": {"message": "Error code: 402 - insufficient credits"}}}])
    assert not bench.credit_failure([{"type": "model.responded", "payload": {"error": {"message": "402"}}}])
    assert "在途请求" in bench.credit_stop_reason([{"type": "run.failed", "payload": {"error": {"message": "402 in_flight_budget_exhausted"}}}])


def test_run_skips_scored_keeps_credit_failed_attempt_and_hides_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUT", tmp_path / "out")
    monkeypatch.setattr(bench, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(bench, "GROUND_TRUTH", tmp_path / "gt.json")
    monkeypatch.setattr(bench, "MANIFEST", tmp_path / "manifest.json")
    (tmp_path / "gt.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    config = {"provider": "openrouter", "model": bench.DEFAULT_MODEL, "base_url": "https://openrouter.ai/api/v1"}
    calls = []

    class FakeWeb:
        def __init__(self, base):
            pass

        def call(self, method, path, **kwargs):
            calls.append(path)
            if path == "/api/bootstrap":
                return {"runtime_default_model_config": config}
            if path == "/api/agent/threads":
                return {"thread_id": "new-thread"}
            if path.endswith("/messages"):
                body = json.dumps(kwargs["json"])
                assert "expected" not in body and "BOF-" not in body and "ground_truth" not in body
                assert kwargs["json"]["target_files"] == ["src/a.c"]
                return {"run_id": "new-run"}
            if path.endswith("/events"):
                return {"events": [{"type": "run.failed", "payload": {"error": {"message": "402 insufficient credits"}}}]}
            return {"status": "failed", "runtime_model_config": config}

    monkeypatch.setattr(bench, "WebClient", FakeWeb)
    monkeypatch.setattr(bench, "check_credits", lambda: {"effective_remaining_usd": 0.01, "checked_utc": "t"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    targets = bench.ordered_targets(GT)[:1]
    checkpoint = {"samples": {"src/a.c": {"status": "credit_exhausted", "attempts": [{"thread_id": "old", "run_id": "old-run"}]}}, "credits": []}
    bench.run(checkpoint, GT, targets, "http://127.0.0.1:7860", bench.DEFAULT_MODEL)
    rec = checkpoint["samples"]["src/a.c"]
    assert len(rec["attempts"]) == 2 and rec["status"] == "credit_exhausted" and "prediction" not in rec
    assert "test-secret" not in (tmp_path / "out/checkpoint.json").read_text()
    rec["prediction"] = {"findings": []}
    rec["status"] = "scored"
    rec["rows"] = []
    calls.clear()
    bench.run(checkpoint, GT, targets, "http://127.0.0.1:7860", bench.DEFAULT_MODEL)
    assert calls == ["/api/bootstrap"]


def test_protocol_change_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUT", tmp_path)
    monkeypatch.setattr(bench, "GROUND_TRUTH", tmp_path / "gt.json")
    monkeypatch.setattr(bench, "MANIFEST", tmp_path / "manifest.json")
    (tmp_path / "gt.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    config = {"provider": "openrouter", "model": bench.DEFAULT_MODEL, "base_url": "https://openrouter.ai/api/v1"}

    class FakeWeb:
        def __init__(self, base):
            pass

        def call(self, method, path, **kwargs):
            return {"runtime_default_model_config": config}

    monkeypatch.setattr(bench, "WebClient", FakeWeb)
    with pytest.raises(ValueError, match="protocol changed"):
        bench.run({"samples": {}, "protocol": {"budget": "other"}}, GT, [], "http://127.0.0.1:7860", bench.DEFAULT_MODEL)


def test_client_refuses_external_backend():
    with pytest.raises(ValueError):
        bench.WebClient("https://example.com")
